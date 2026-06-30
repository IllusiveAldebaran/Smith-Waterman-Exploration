#!/usr/bin/env python3
"""Explore Smith-Waterman DP and Farrar's striped method.

This file intentionally ignores AIE/MLIR. It is a readable, instrumented Python
reference for answering questions like:

* How many DP cells does ordinary Smith-Waterman fill?
* How many traceback steps/directions occur?
* How often does Farrar's method call each conceptual internal step?
* How much relative time is spent in DP fill, traceback, profile build,
  Farrar's main striped pass, and Farrar's lazy-F correction?

The implementation favors clarity and traceability over speed. Every major
stage records both occurrence counts and elapsed time.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass
from math import ceil
from pathlib import Path
import random
from time import perf_counter


NEG_INF = -10**9


@dataclass(frozen=True)
class AlignmentResult:
    score: int
    end_query: int
    end_reference: int


@dataclass(frozen=True)
class TracebackResult:
    start_query: int
    start_reference: int
    aligned_query: str
    aligned_reference: str
    path: tuple[str, ...]


@dataclass
class GridCell:
    query_len: int
    reference_len: int
    pairs: int = 0
    score_sum: int = 0
    mismatches: int = 0
    counts: Counter[str] | None = None
    times: Counter[str] | None = None

    def __post_init__(self) -> None:
        if self.counts is None:
            self.counts = Counter()
        if self.times is None:
            self.times = Counter()

    def add_run(
        self,
        farrar_score: int,
        counts: Counter[str],
        times: Counter[str],
        score_mismatch: bool,
    ) -> None:
        self.pairs += 1
        self.score_sum += farrar_score
        self.mismatches += int(score_mismatch)
        assert self.counts is not None
        assert self.times is not None
        self.counts.update(counts)
        self.times.update(times)

    def count_total(self, key: str) -> float:
        assert self.counts is not None
        return float(self.counts.get(key, 0))

    def time_total(self, key: str) -> float:
        assert self.times is not None
        return float(self.times.get(key, 0.0))

    def metric_total(self, key: str) -> float:
        if key.startswith("time:"):
            return self.time_total(key.removeprefix("time:"))
        if key == "score_pass_rate":
            return self.count_total("score_passes_min")
        return self.count_total(key)

    def metric_average(self, key: str) -> float:
        if key == "score_pass_rate":
            return self.count_total("score_passes_min") / self.pairs if self.pairs else 0.0
        return self.metric_total(key) / self.pairs if self.pairs else 0.0


class Recorder:
    """Collect occurrence counts and stage timings.

    The counters are deliberately named after algorithmic events rather than
    low-level Python operations. That keeps the output close to the question:
    "how many times does this part of DP/Farrar get called?"
    """

    def __init__(self, verbose: int = 1) -> None:
        self.counts: Counter[str] = Counter()
        self.times: Counter[str] = Counter()
        self.verbose = verbose
        self.events: list[str] = []

    def count(self, name: str, amount: int = 1) -> None:
        self.counts[name] += amount
        if self.verbose >= 3:
            self.events.append(f"count {name} += {amount}")

    @contextmanager
    def timed(self, name: str):
        if self.verbose >= 2:
            self.events.append(f"start {name}")
        start = perf_counter()
        try:
            yield
        finally:
            elapsed = perf_counter() - start
            self.times[name] += elapsed
            if self.verbose >= 2:
                self.events.append(f"end {name}: {elapsed:.9f}s")


def normalize_sequence(sequence: str, label: str) -> str:
    normalized = "".join(sequence.split()).upper()
    if not normalized:
        raise ValueError(f"{label} sequence is empty")
    return normalized


def read_sequence(value: str | None, file_value: str | None, label: str) -> str:
    if value is not None and file_value is not None:
        raise ValueError(f"use either --{label} or --{label}-file, not both")
    if file_value is None:
        if value is None:
            raise ValueError(f"missing --{label}")
        return normalize_sequence(value, label)

    text = Path(file_value).expanduser().read_text(encoding="utf-8")
    lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith(">"):
            lines.append(stripped)
    return normalize_sequence("".join(lines), label)


def score_pair(a: str, b: str, match: int, mismatch: int) -> int:
    return match if a == b else mismatch


def smith_waterman_dp(
    query: str,
    reference: str,
    match: int,
    mismatch: int,
    gap_open: int,
    gap_extend: int,
    rec: Recorder,
) -> tuple[AlignmentResult, list[list[int]], list[list[str]]]:
    """Fill the scalar Smith-Waterman affine-gap DP matrices.

    H is the best local score ending at each cell.
    E is the best score ending with a horizontal gap.
    F is the best score ending with a vertical gap.
    ptr records the selected predecessor for traceback.
    """

    rec.count("smith_waterman.invocations")
    with rec.timed("smith_waterman.dp_fill"):
        m = len(query)
        n = len(reference)
        h = [[0] * (n + 1) for _ in range(m + 1)]
        e = [[NEG_INF] * (n + 1) for _ in range(m + 1)]
        f = [[NEG_INF] * (n + 1) for _ in range(m + 1)]
        ptr = [["stop"] * (n + 1) for _ in range(m + 1)]

        best = AlignmentResult(0, 0, 0)
        for i in range(1, m + 1):
            rec.count("smith_waterman.dp_rows")
            for j in range(1, n + 1):
                rec.count("smith_waterman.dp_cells")
                rec.count("smith_waterman.substitution_scores")

                rec.count("smith_waterman.gap_e_updates")
                e[i][j] = max(h[i][j - 1] - gap_open, e[i][j - 1] - gap_extend)

                rec.count("smith_waterman.gap_f_updates")
                f[i][j] = max(h[i - 1][j] - gap_open, f[i - 1][j] - gap_extend)

                rec.count("smith_waterman.diagonal_updates")
                diag = h[i - 1][j - 1] + score_pair(
                    query[i - 1], reference[j - 1], match, mismatch
                )

                rec.count("smith_waterman.cell_max_reductions")
                h[i][j] = max(0, diag, e[i][j], f[i][j])

                if h[i][j] == 0:
                    ptr[i][j] = "stop"
                elif h[i][j] == diag:
                    ptr[i][j] = "diag"
                elif h[i][j] == e[i][j]:
                    ptr[i][j] = "left"
                else:
                    ptr[i][j] = "up"

                if h[i][j] > best.score:
                    rec.count("smith_waterman.best_score_updates")
                    best = AlignmentResult(h[i][j], i, j)

    return best, h, ptr


def traceback_alignment(
    query: str,
    reference: str,
    h: list[list[int]],
    ptr: list[list[str]],
    end: AlignmentResult,
    rec: Recorder,
) -> TracebackResult:
    """Backtrack from the best local-alignment cell to the zero boundary."""

    with rec.timed("smith_waterman.traceback"):
        i = end.end_query
        j = end.end_reference
        aligned_query = []
        aligned_reference = []
        path = []

        while i > 0 and j > 0 and h[i][j] > 0:
            direction = ptr[i][j]
            rec.count("smith_waterman.backtrack_steps")
            rec.count(f"smith_waterman.backtrack_{direction}")
            path.append(direction)

            if direction == "diag":
                aligned_query.append(query[i - 1])
                aligned_reference.append(reference[j - 1])
                i -= 1
                j -= 1
            elif direction == "left":
                aligned_query.append("-")
                aligned_reference.append(reference[j - 1])
                j -= 1
            elif direction == "up":
                aligned_query.append(query[i - 1])
                aligned_reference.append("-")
                i -= 1
            else:
                break

        rec.count("smith_waterman.backtrack_stop")

    return TracebackResult(
        start_query=i + 1,
        start_reference=j + 1,
        aligned_query="".join(reversed(aligned_query)),
        aligned_reference="".join(reversed(aligned_reference)),
        path=tuple(reversed(path)),
    )


def striped_index_to_query_index(segment: int, lane: int, seg_len: int) -> int:
    return lane * seg_len + segment


def make_query_profile(
    query: str,
    alphabet: list[str],
    lanes: int,
    match: int,
    mismatch: int,
    rec: Recorder,
) -> tuple[dict[str, list[list[int]]], int]:
    """Build Farrar's query profile.

    Farrar's striped method rearranges the query into vectors. For each possible
    reference symbol, the profile stores the substitution score for every query
    lane. This moves substitution lookup out of the inner DP recurrence.
    """

    with rec.timed("farrar.profile_build"):
        seg_len = ceil(len(query) / lanes)
        profile: dict[str, list[list[int]]] = {}

        for ch in alphabet:
            rec.count("farrar.profile_symbols")
            vectors = []
            for segment in range(seg_len):
                rec.count("farrar.profile_vectors")
                vec = []
                for lane in range(lanes):
                    rec.count("farrar.profile_lanes")
                    q_index = striped_index_to_query_index(segment, lane, seg_len)
                    if q_index < len(query):
                        vec.append(score_pair(query[q_index], ch, match, mismatch))
                    else:
                        vec.append(NEG_INF)
                vectors.append(vec)
            profile[ch] = vectors

    return profile, seg_len


def vector_add(a: list[int], b: list[int], rec: Recorder, label: str) -> list[int]:
    rec.count(f"{label}.vector_add_calls")
    rec.count(f"{label}.lane_adds", len(a))
    return [x + y for x, y in zip(a, b)]


def vector_sub_scalar(
    a: list[int], value: int, rec: Recorder, label: str
) -> list[int]:
    rec.count(f"{label}.vector_sub_scalar_calls")
    rec.count(f"{label}.lane_subs", len(a))
    return [x - value for x in a]


def vector_max(a: list[int], b: list[int], rec: Recorder, label: str) -> list[int]:
    rec.count(f"{label}.vector_max_calls")
    rec.count(f"{label}.lane_maxes", len(a))
    return [max(x, y) for x, y in zip(a, b)]


def vector_max_scalar(
    a: list[int], value: int, rec: Recorder, label: str
) -> list[int]:
    rec.count(f"{label}.vector_max_scalar_calls")
    rec.count(f"{label}.lane_maxes", len(a))
    return [max(x, value) for x in a]


def shift_one_lane(v: list[int], insert: int, rec: Recorder, label: str) -> list[int]:
    rec.count(f"{label}.lane_shift_calls")
    return [insert] + v[:-1]


def farrar_striped_sw(
    query: str,
    reference: str,
    match: int,
    mismatch: int,
    gap_open: int,
    gap_extend: int,
    lanes: int,
    rec: Recorder,
) -> AlignmentResult:
    """Compute Smith-Waterman score using Farrar's striped method.

    The major Farrar steps recorded here are:

    1. Query profile build.
    2. Per-reference-symbol main striped DP pass.
    3. Lazy-F correction pass, which propagates vertical-gap scores across
       vector lanes until no lane needs further correction.

    Endpoint tie-breaking may differ from scalar DP, but the best score should
    match.
    """

    rec.count("farrar.invocations")
    if lanes < 1:
        raise ValueError("lanes must be >= 1")
    if not query or not reference:
        return AlignmentResult(0, 0, 0)

    alphabet = sorted(set(query) | set(reference))
    profile, seg_len = make_query_profile(query, alphabet, lanes, match, mismatch, rec)
    zero = [0] * lanes
    neg = [NEG_INF] * lanes
    h_store = [zero[:] for _ in range(seg_len)]
    e_store = [neg[:] for _ in range(seg_len)]
    best = AlignmentResult(0, 0, 0)

    for reference_pos, reference_ch in enumerate(reference, start=1):
        rec.count("farrar.reference_columns")
        h_load = [v[:] for v in h_store]
        f = neg[:]

        # Step A: Shift the previous column's last vector to form the
        # diagonal dependency for the first segment of this column.
        h = shift_one_lane(h_load[-1], 0, rec, "farrar.column_start")

        # Step B: Main striped pass. This computes H/E/F in vector segments
        # using the precomputed query profile for the current reference char.
        with rec.timed("farrar.main_striped_pass"):
            for segment in range(seg_len):
                rec.count("farrar.main_segment_iterations")
                rec.count("farrar.profile_vector_loads")

                h = vector_add(
                    h,
                    profile[reference_ch][segment],
                    rec,
                    "farrar.main.h_plus_profile",
                )
                h = vector_max(h, e_store[segment], rec, "farrar.main.max_h_e")
                h = vector_max(h, f, rec, "farrar.main.max_h_f")
                h = vector_max_scalar(h, 0, rec, "farrar.main.max_h_zero")
                h_store[segment] = h

                for lane, score in enumerate(h):
                    q_index = striped_index_to_query_index(segment, lane, seg_len)
                    if q_index < len(query) and score > best.score:
                        rec.count("farrar.best_score_updates")
                        best = AlignmentResult(score, q_index + 1, reference_pos)

                h_gap = vector_sub_scalar(h, gap_open, rec, "farrar.main.h_gap")
                e_ext = vector_sub_scalar(
                    e_store[segment], gap_extend, rec, "farrar.main.e_ext"
                )
                e_store[segment] = vector_max(
                    e_ext, h_gap, rec, "farrar.main.max_e"
                )

                f_ext = vector_sub_scalar(f, gap_extend, rec, "farrar.main.f_ext")
                f = vector_max(f_ext, h_gap, rec, "farrar.main.max_f")

                # The next segment's diagonal dependency comes from the
                # stored H value from the previous reference column.
                h = h_load[segment]

        # Step C: Lazy-F correction. The main pass computes many F values,
        # but vertical gaps can need propagation across striped lanes. This
        # loop keeps shifting F and correcting H until all lanes are stable.
        with rec.timed("farrar.lazy_f_correction"):
            for _ in range(lanes):
                rec.count("farrar.lazy_f_lane_passes")
                f = shift_one_lane(f, NEG_INF, rec, "farrar.lazy_f.shift")
                stop = True

                for segment in range(seg_len):
                    rec.count("farrar.lazy_f_segment_iterations")
                    h = h_store[segment]
                    corrected = vector_max(h, f, rec, "farrar.lazy_f.max_h_f")
                    h_store[segment] = corrected

                    for lane, score in enumerate(corrected):
                        q_index = striped_index_to_query_index(segment, lane, seg_len)
                        if q_index < len(query) and score > best.score:
                            rec.count("farrar.best_score_updates")
                            best = AlignmentResult(score, q_index + 1, reference_pos)

                    h_gap = vector_sub_scalar(
                        corrected, gap_open, rec, "farrar.lazy_f.h_gap"
                    )
                    f = vector_sub_scalar(f, gap_extend, rec, "farrar.lazy_f.f_ext")

                    rec.count("farrar.lazy_f_stop_tests")
                    rec.count("farrar.lazy_f_stop_test_lanes", lanes)
                    if any(
                        f_lane > h_gap_lane for f_lane, h_gap_lane in zip(f, h_gap)
                    ):
                        stop = False

                if stop:
                    rec.count("farrar.lazy_f_early_exits")
                    break

    return best


def format_matrix(matrix: list[list[int]], query: str, reference: str) -> str:
    header = "      " + " ".join(f"{ch:>3}" for ch in "-" + reference)
    lines = [header]
    for i, row in enumerate(matrix):
        label = "-" if i == 0 else query[i - 1]
        lines.append(f"{label:>3}   " + " ".join(f"{cell:>3}" for cell in row))
    return "\n".join(lines)


def print_counts(title: str, counts: Counter[str], prefix: str | None = None) -> None:
    print(f"\n{title}:")
    for key in sorted(counts):
        if prefix is None or key.startswith(prefix):
            print(f"  {key}: {counts[key]}")


def print_times(times: Counter[str]) -> None:
    total = sum(times.values())
    print("\nTiming:")
    for key in sorted(times):
        seconds = times[key]
        pct = (seconds / total * 100.0) if total else 0.0
        print(f"  {key}: {seconds:.6f}s ({pct:.2f}% of recorded time)")


def smith_waterman_time(times: Counter[str]) -> float:
    return times.get("smith_waterman.dp_fill", 0.0) + times.get(
        "smith_waterman.traceback", 0.0
    )


def farrar_time(times: Counter[str]) -> float:
    return (
        times.get("farrar.profile_build", 0.0)
        + times.get("farrar.main_striped_pass", 0.0)
        + times.get("farrar.lazy_f_correction", 0.0)
    )


def print_event_log(rec: Recorder) -> None:
    if not rec.events:
        return
    print("\nEvent log:")
    for event in rec.events:
        print(f"  {event}")


def parse_lengths(spec: str) -> list[int]:
    """Parse length specs such as '8,16,32' or '8:64:8'."""

    values: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" in part:
            pieces = [int(x) for x in part.split(":")]
            if len(pieces) == 2:
                start, stop = pieces
                step = 1
            elif len(pieces) == 3:
                start, stop, step = pieces
            else:
                raise ValueError(f"bad length range: {part}")
            if step <= 0:
                raise ValueError("length range step must be positive")
            values.extend(range(start, stop + 1, step))
        else:
            values.append(int(part))

    unique = sorted(set(values))
    if not unique or any(v < 1 for v in unique):
        raise ValueError("lengths must contain positive integers")
    return unique


def random_sequence(length: int, alphabet: str, rng: random.Random) -> str:
    return "".join(rng.choice(alphabet) for _ in range(length))


def run_one_pair(
    query: str,
    reference: str,
    args: argparse.Namespace,
    validate_scalar: bool,
) -> tuple[AlignmentResult, bool, Counter[str], Counter[str], list[str]]:
    rec = Recorder(verbose=args.verbose)
    score_mismatch = False
    scalar_score: int | None = None
    if validate_scalar:
        scalar, matrix, ptr = smith_waterman_dp(
            query,
            reference,
            args.match,
            args.mismatch,
            args.gap_open,
            args.gap_extend,
            rec,
        )
        traceback_alignment(query, reference, matrix, ptr, scalar, rec)
        scalar_score = scalar.score

    farrar = farrar_striped_sw(
        query,
        reference,
        args.match,
        args.mismatch,
        args.gap_open,
        args.gap_extend,
        args.lanes,
        rec,
    )
    if scalar_score is not None:
        score_mismatch = scalar_score != farrar.score
    return farrar, score_mismatch, rec.counts, rec.times, rec.events


def grid_rows(
    query_lengths: list[int],
    reference_lengths: list[int],
    args: argparse.Namespace,
) -> list[GridCell]:
    rng = random.Random(args.seed)
    cells: list[GridCell] = []
    total_cells = len(query_lengths) * len(reference_lengths)
    cell_index = 0

    for reference_len in reference_lengths:
        for query_len in query_lengths:
            cell_index += 1
            cell = GridCell(query_len=query_len, reference_len=reference_len)
            if args.progress:
                print(
                    f"grid cell {cell_index}/{total_cells}: "
                    f"query_len={query_len}, reference_len={reference_len}",
                    flush=True,
                )

            for _ in range(args.pairs_per_cell):
                query = random_sequence(query_len, args.alphabet, rng)
                reference = random_sequence(reference_len, args.alphabet, rng)
                farrar, mismatch, counts, times, events = run_one_pair(
                    query, reference, args, args.grid_validate_scalar
                )
                if args.min_score is not None:
                    counts = counts.copy()
                    passed = int(farrar.score >= args.min_score)
                    counts["score_passes_min"] = passed
                    counts["score_fails_min"] = 1 - passed
                cell.add_run(farrar.score, counts, times, mismatch)
                if args.verbose >= 2:
                    trace_label = (
                        "full event trace"
                        if args.verbose >= 3
                        else "stage event trace"
                    )
                    print(
                        f"\n{trace_label} for pair "
                        f"query_len={query_len}, reference_len={reference_len}:"
                    )
                    for event in events:
                        print(f"  {event}")
            cells.append(cell)

    return cells


def print_grid_summary(
    cells: list[GridCell], verbose: int, min_score: int | None
) -> None:
    total_pairs = sum(c.pairs for c in cells)
    total_counts: Counter[str] = Counter()
    total_times: Counter[str] = Counter()
    for cell in cells:
        assert cell.counts is not None
        assert cell.times is not None
        total_counts.update(cell.counts)
        total_times.update(cell.times)

    sw_invocations = total_counts.get("smith_waterman.invocations", 0)
    farrar_invocations = total_counts.get("farrar.invocations", 0)
    min_score_passes = total_counts.get("score_passes_min", 0)
    min_score_pass_rate = min_score_passes / total_pairs if total_pairs else 0.0
    sw_seconds = smith_waterman_time(total_times)
    farrar_seconds = farrar_time(total_times)

    summary = (
        "grid summary: "
        f"pairs={total_pairs}, "
        f"smith_waterman_invocations={sw_invocations}, "
        f"smith_waterman_time={sw_seconds:.6f}s, "
        f"farrar_invocations={farrar_invocations}, "
        f"farrar_time={farrar_seconds:.6f}s"
    )
    if min_score is not None:
        summary += (
            f", min_score={min_score}, "
            f"min_score_passes={min_score_passes}, "
            f"min_score_pass_rate={min_score_pass_rate:.6f}"
        )
    print(summary)

    if verbose >= 2:
        print("\nper-cell debug metrics:")
        for cell in cells:
            assert cell.counts is not None
            assert cell.times is not None
            print(
                "  "
                f"query_len={cell.query_len}, "
                f"reference_len={cell.reference_len}, "
                f"pairs={cell.pairs}, "
                f"sw_calls={cell.count_total('smith_waterman.invocations'):.0f}, "
                f"sw_time={smith_waterman_time(cell.times):.6f}s, "
                f"farrar_calls={cell.count_total('farrar.invocations'):.0f}, "
                f"farrar_time={farrar_time(cell.times):.6f}s, "
                f"min_score_passes={cell.count_total('score_passes_min'):.0f}, "
                f"min_score_pass_rate={cell.metric_average('score_pass_rate'):.6f}, "
                f"lazy_f_passes={cell.count_total('farrar.lazy_f_lane_passes'):.0f}, "
                f"main_segments={cell.count_total('farrar.main_segment_iterations'):.0f}"
            )

    if verbose >= 3:
        print_counts("Full grid counts", total_counts)
        print_times(total_times)


def write_grid_csv(
    path: str, cells: list[GridCell], metric: str, min_score: int | None
) -> None:
    headers = [
        "query_len",
        "reference_len",
        "pairs",
        "avg_score",
        "mismatches",
        "min_score",
        "min_score_passes",
        "min_score_pass_rate",
        "metric",
        "metric_total",
        "metric_avg",
        "profile_build_s_total",
        "main_striped_pass_s_total",
        "lazy_f_correction_s_total",
        "profile_build_s_avg",
        "main_striped_pass_s_avg",
        "lazy_f_correction_s_avg",
        "lazy_f_lane_passes_total",
        "lazy_f_segment_iterations_total",
        "lazy_f_max_h_f_calls_total",
        "main_segment_iterations_total",
    ]
    with Path(path).expanduser().open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for cell in cells:
            pairs = cell.pairs or 1
            writer.writerow(
                {
                    "query_len": cell.query_len,
                    "reference_len": cell.reference_len,
                    "pairs": cell.pairs,
                    "avg_score": cell.score_sum / pairs,
                    "mismatches": cell.mismatches,
                    "min_score": "" if min_score is None else min_score,
                    "min_score_passes": cell.count_total("score_passes_min"),
                    "min_score_pass_rate": cell.count_total("score_passes_min")
                    / pairs,
                    "metric": metric,
                    "metric_total": cell.metric_total(metric),
                    "metric_avg": cell.metric_average(metric),
                    "profile_build_s_total": cell.time_total("farrar.profile_build"),
                    "main_striped_pass_s_total": cell.time_total(
                        "farrar.main_striped_pass"
                    ),
                    "lazy_f_correction_s_total": cell.time_total(
                        "farrar.lazy_f_correction"
                    ),
                    "profile_build_s_avg": cell.time_total("farrar.profile_build")
                    / pairs,
                    "main_striped_pass_s_avg": cell.time_total(
                        "farrar.main_striped_pass"
                    )
                    / pairs,
                    "lazy_f_correction_s_avg": cell.time_total(
                        "farrar.lazy_f_correction"
                    )
                    / pairs,
                    "lazy_f_lane_passes_total": cell.count_total(
                        "farrar.lazy_f_lane_passes"
                    ),
                    "lazy_f_segment_iterations_total": cell.count_total(
                        "farrar.lazy_f_segment_iterations"
                    ),
                    "lazy_f_max_h_f_calls_total": cell.count_total(
                        "farrar.lazy_f.max_h_f.vector_max_calls"
                    ),
                    "main_segment_iterations_total": cell.count_total(
                        "farrar.main_segment_iterations"
                    ),
                }
            )


def heat_color(value: float, min_value: float, max_value: float) -> str:
    if max_value <= min_value:
        t = 0.0
    else:
        t = (value - min_value) / (max_value - min_value)
    t = max(0.0, min(1.0, t))
    # White -> orange -> red, dependency-free and readable in SVG.
    if t < 0.5:
        local = t / 0.5
        r = 255
        g = round(255 - 90 * local)
        b = round(255 - 205 * local)
    else:
        local = (t - 0.5) / 0.5
        r = 255
        g = round(165 - 145 * local)
        b = round(50 - 50 * local)
    return f"#{r:02x}{g:02x}{b:02x}"


def write_heatmap_svg(
    path: str,
    cells: list[GridCell],
    query_lengths: list[int],
    reference_lengths: list[int],
    metric: str,
) -> None:
    cell_size = 34
    left = 90
    top = 70
    right = 30
    bottom = 60
    width = left + len(reference_lengths) * cell_size + right
    height = top + len(query_lengths) * cell_size + bottom
    cell_by_coord = {(c.query_len, c.reference_len): c for c in cells}
    values = [c.metric_average(metric) for c in cells]
    min_value = min(values) if values else 0.0
    max_value = max(values) if values else 0.0

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width / 2:.1f}" y="24" text-anchor="middle" font-family="monospace" font-size="14">Farrar heatmap: avg {metric}</text>',
        f'<text x="{width / 2:.1f}" y="44" text-anchor="middle" font-family="monospace" font-size="11">min={min_value:.6g}, max={max_value:.6g}</text>',
        f'<text x="{left + len(reference_lengths) * cell_size / 2:.1f}" y="{height - 18}" text-anchor="middle" font-family="monospace" font-size="12">reference length</text>',
        f'<text x="18" y="{top + len(query_lengths) * cell_size / 2:.1f}" transform="rotate(-90 18 {top + len(query_lengths) * cell_size / 2:.1f})" text-anchor="middle" font-family="monospace" font-size="12">query length</text>',
    ]

    for x, reference_len in enumerate(reference_lengths):
        cx = left + x * cell_size + cell_size / 2
        lines.append(
            f'<text x="{cx:.1f}" y="{top - 10}" text-anchor="middle" font-family="monospace" font-size="10">{reference_len}</text>'
        )

    for y, query_len in enumerate(query_lengths):
        cy = top + y * cell_size + cell_size / 2 + 4
        lines.append(
            f'<text x="{left - 10}" y="{cy:.1f}" text-anchor="end" font-family="monospace" font-size="10">{query_len}</text>'
        )

    for y, query_len in enumerate(query_lengths):
        for x, reference_len in enumerate(reference_lengths):
            cell = cell_by_coord[(query_len, reference_len)]
            value = cell.metric_average(metric)
            color = heat_color(value, min_value, max_value)
            rx = left + x * cell_size
            ry = top + y * cell_size
            lines.append(
                f'<rect x="{rx}" y="{ry}" width="{cell_size}" height="{cell_size}" fill="{color}" stroke="#333" stroke-width="0.5"/>'
            )
            lines.append(
                f'<title>query_len={query_len}, reference_len={reference_len}, avg={value:.6g}, pairs={cell.pairs}</title>'
            )

    lines.append("</svg>")
    Path(path).expanduser().write_text("\n".join(lines) + "\n", encoding="utf-8")


def unnumbered_stem(path: Path) -> str:
    stem = path.stem
    while stem and stem[-1].isdigit():
        stem = stem[:-1]
    return stem or path.stem


def numbered_path(path: str, run_number: int) -> str:
    expanded = Path(path).expanduser()
    stem = unnumbered_stem(expanded)
    return str(expanded.with_name(f"{stem}{run_number}{expanded.suffix}"))


def next_grid_output_paths(
    grid_csv: str, heatmap_svg: str | None
) -> tuple[str, str | None]:
    run_number = 1
    while True:
        csv_path = numbered_path(grid_csv, run_number)
        svg_path = numbered_path(heatmap_svg, run_number) if heatmap_svg else None
        csv_exists = Path(csv_path).exists()
        svg_exists = Path(svg_path).exists() if svg_path else False
        if not csv_exists and not svg_exists:
            return csv_path, svg_path
        run_number += 1


def resolve_heatmap_metric(args: argparse.Namespace) -> str:
    if args.heatmap_metric is not None:
        return args.heatmap_metric
    if args.min_score is not None:
        return "score_pass_rate"
    return "farrar.lazy_f_lane_passes"


def run_grid_mode(args: argparse.Namespace) -> None:
    query_lengths = parse_lengths(args.query_lengths)
    reference_lengths = parse_lengths(args.reference_lengths)
    grid_csv, heatmap_svg = next_grid_output_paths(args.grid_csv, args.heatmap_svg)
    heatmap_metric = resolve_heatmap_metric(args)
    cells = grid_rows(query_lengths, reference_lengths, args)
    write_grid_csv(grid_csv, cells, heatmap_metric, args.min_score)
    if heatmap_svg:
        write_heatmap_svg(
            heatmap_svg,
            cells,
            query_lengths,
            reference_lengths,
            heatmap_metric,
        )

    print_grid_summary(cells, args.verbose, args.min_score)
    if args.verbose >= 2:
        print(f"csv: {grid_csv}")
        if heatmap_svg:
            print(f"svg: {heatmap_svg}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", default=None)
    parser.add_argument("--reference", "--target", dest="reference", default=None)
    parser.add_argument("--query-file", default=None)
    parser.add_argument("--reference-file", "--target-file", dest="reference_file")
    parser.add_argument("--match", type=int, default=2)
    parser.add_argument("--mismatch", type=int, default=-1)
    parser.add_argument("--gap-open", type=int, default=3)
    parser.add_argument("--gap-extend", type=int, default=1)
    parser.add_argument("--lanes", type=int, default=8)
    parser.add_argument("--matrix", action="store_true")
    parser.add_argument(
        "--verbose",
        type=int,
        choices=[1, 2, 3],
        default=1,
        help=(
            "1: compact output, 2: debug metrics/stage events, "
            "3: full counters and event log"
        ),
    )
    parser.add_argument(
        "--grid",
        action="store_true",
        help="run random sequence pairs over a query/reference length grid",
    )
    parser.add_argument(
        "--query-lengths",
        default="8:64:8",
        help="grid query lengths, e.g. '8,16,32' or '8:128:8'",
    )
    parser.add_argument(
        "--reference-lengths",
        default="8:64:8",
        help="grid reference lengths, e.g. '8,16,32' or '8:128:8'",
    )
    parser.add_argument("--pairs-per-cell", type=int, default=100)
    parser.add_argument("--alphabet", default="ACGT")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--min-score",
        type=int,
        default=None,
        help=(
            "minimum Farrar score to count as passing; with grid mode, the "
            "default heatmap metric becomes score_pass_rate"
        ),
    )
    parser.add_argument(
        "--grid-csv",
        default="farrar_grid.csv",
        help="CSV filename template; grid mode writes the next numbered path",
    )
    parser.add_argument(
        "--heatmap-svg",
        default="farrar_heatmap.svg",
        help="SVG filename template; grid mode writes the next numbered path",
    )
    parser.add_argument(
        "--heatmap-metric",
        default=None,
        help=(
            "count key or time key to heatmap. Time keys use 'time:' prefix, "
            "e.g. time:farrar.lazy_f_correction. Defaults to score_pass_rate "
            "when --min-score is set, otherwise farrar.lazy_f_lane_passes"
        ),
    )
    parser.add_argument(
        "--grid-validate-scalar",
        action="store_true",
        help="also run scalar DP for each pair and record score mismatches",
    )
    parser.add_argument("--progress", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.grid:
        run_grid_mode(args)
        return

    if args.query is None and args.query_file is None:
        args.query = "ACACACTA"
    if args.reference is None and args.reference_file is None:
        args.reference = "AGCACACA"

    query = read_sequence(args.query, args.query_file, "query")
    reference = read_sequence(args.reference, args.reference_file, "reference")

    rec = Recorder(verbose=args.verbose)
    scalar, matrix, ptr = smith_waterman_dp(
        query,
        reference,
        args.match,
        args.mismatch,
        args.gap_open,
        args.gap_extend,
        rec,
    )
    traceback = traceback_alignment(query, reference, matrix, ptr, scalar, rec)
    farrar = farrar_striped_sw(
        query,
        reference,
        args.match,
        args.mismatch,
        args.gap_open,
        args.gap_extend,
        args.lanes,
        rec,
    )

    if scalar.score != farrar.score:
        raise SystemExit(
            f"score mismatch: scalar={scalar.score}, farrar={farrar.score}"
        )

    print(
        "single-pair summary: "
        f"smith_waterman_invocations={rec.counts.get('smith_waterman.invocations', 0)}, "
        f"smith_waterman_time={smith_waterman_time(rec.times):.6f}s, "
        f"farrar_invocations={rec.counts.get('farrar.invocations', 0)}, "
        f"farrar_time={farrar_time(rec.times):.6f}s"
    )

    if args.verbose >= 2:
        print("\nSmith-Waterman scalar result:")
        print(
            f"  score={scalar.score} "
            f"end_query={scalar.end_query} "
            f"end_reference={scalar.end_reference}"
        )
        print(
            f"  start_query={traceback.start_query} "
            f"start_reference={traceback.start_reference}"
        )
        print(f"  aligned_query    ={traceback.aligned_query}")
        print(f"  aligned_reference={traceback.aligned_reference}")
        print("Farrar striped result:")
        print(
            f"  score={farrar.score} "
            f"end_query={farrar.end_query} "
            f"end_reference={farrar.end_reference}"
        )
        print_times(rec.times)
        print_event_log(rec)

    if args.verbose >= 3:
        print_counts("Smith-Waterman DP counts", rec.counts, "smith_waterman.")
        print_counts("Farrar counts", rec.counts, "farrar.")

    if args.matrix and args.verbose >= 3:
        print("\nSmith-Waterman H matrix:")
        print(format_matrix(matrix, query, reference))


if __name__ == "__main__":
    main()

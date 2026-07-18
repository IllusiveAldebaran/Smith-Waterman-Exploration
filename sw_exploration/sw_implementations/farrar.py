"""Farrar's striped Smith-Waterman method with full instrumentation.

The algorithm rearranges query residues into SIMD-style vector segments so
that every reference column can be processed with data-parallel vector ops.
Three conceptual stages are recorded separately:
  1. Query profile build  – precomputes substitution scores per (symbol, segment).
  2. Main striped pass    – per-reference-column H/E/F update in segment order.
  3. Lazy-F correction    – propagates vertical-gap scores across lane boundaries
                            until no further correction is needed.

pen layout: array.array('b', [match, mismatch, del_open, del_ext, ins_open, ins_ext])
  del_open/del_ext apply to E (horizontal gaps).
  ins_open/ins_ext apply to F (vertical gaps).
  match/mismatch are pre-negated; score_pair() in scalar.py negates them back
  to an actual score delta.

FarrarImpl is the registry entry point. lanes is set at construction time.
"""

from __future__ import annotations

import array
from math import ceil

from .scalar import score_pair
from ..types import Aligner, AlignmentResult, Recorder, NEG_INF


def striped_index_to_query_index(segment: int, lane: int, seg_len: int) -> int:
    """Convert a (segment, lane) pair back to a linear query index."""
    return lane * seg_len + segment


def make_query_profile(
    query: str,
    reference: str,
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

        for ch in sorted(set(reference)):
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


def _run_pair(
    query: str,
    reference: str,
    pen: array.array,
    lanes: int,
    rec: Recorder,
) -> AlignmentResult:
    """Align one pair using Farrar's striped method.

    Records one "h_matrix" cell event per filled cell (H-matrix coordinates,
    (query_len+1) × (ref_len+1) with row 0 and col 0 as the zero DP boundary)
    into rec.
    """
    match, mismatch, del_open, del_ext, ins_open, ins_ext = pen
    rec.count("farrar.invocations")
    if lanes < 1:
        raise ValueError("lanes must be >= 1")

    query_len = len(query)
    ref_len = len(reference)

    if not query or not reference:
        return AlignmentResult(0, 0, 0)

    profile, seg_len = make_query_profile(query, reference, lanes, match, mismatch, rec)
    zero = [0] * lanes
    neg = [NEG_INF] * lanes
    h_store = [zero[:] for _ in range(seg_len)]
    e_store = [neg[:] for _ in range(seg_len)]
    best = AlignmentResult(0, 0, 0)

    with rec.timed("smith_waterman.dp_fill"):
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
                        if q_index < query_len and score > best.score:
                            rec.count("farrar.best_score_updates")
                            best = AlignmentResult(score, q_index + 1, reference_pos)

                    h_del_gap = vector_sub_scalar(h, del_open, rec, "farrar.main.h_del_gap")
                    h_ins_gap = vector_sub_scalar(h, ins_open, rec, "farrar.main.h_ins_gap")

                    e_ext = vector_sub_scalar(
                        e_store[segment], del_ext, rec, "farrar.main.e_ext"
                    )
                    e_store[segment] = vector_max(
                        e_ext, h_del_gap, rec, "farrar.main.max_e"
                    )

                    f_ext = vector_sub_scalar(f, ins_ext, rec, "farrar.main.f_ext")
                    f = vector_max(f_ext, h_ins_gap, rec, "farrar.main.max_f")

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

                        # Record cells where F actually improved H (in H-matrix coords).
                        for lane_k, (orig, corr) in enumerate(zip(h, corrected)):
                            if corr > orig:
                                q_idx = striped_index_to_query_index(segment, lane_k, seg_len)
                                if q_idx < query_len:
                                    rec.add_cell_event(
                                        "farrar.lazy_f_trigger", q_idx + 1, reference_pos
                                    )

                        for lane, score in enumerate(corrected):
                            q_index = striped_index_to_query_index(segment, lane, seg_len)
                            if q_index < query_len and score > best.score:
                                rec.count("farrar.best_score_updates")
                                best = AlignmentResult(score, q_index + 1, reference_pos)

                        h_ins_gap = vector_sub_scalar(
                            corrected, ins_open, rec, "farrar.lazy_f.h_ins_gap"
                        )
                        f = vector_sub_scalar(f, ins_ext, rec, "farrar.lazy_f.f_ext")

                        rec.count("farrar.lazy_f_stop_tests")
                        rec.count("farrar.lazy_f_stop_test_lanes", lanes)
                        if any(
                            f_lane > h_ins_gap_lane
                            for f_lane, h_ins_gap_lane in zip(f, h_ins_gap)
                        ):
                            stop = False

                    if stop:
                        rec.count("farrar.lazy_f_early_exits")
                        break
                    else:
                        # F actually propagated past this iteration — real correction work.
                        rec.count("farrar.lazy_f_corrections")

            # Record final corrected H values for this column as h_matrix cell events.
            for segment in range(seg_len):
                for lane, score in enumerate(h_store[segment]):
                    q_idx = striped_index_to_query_index(segment, lane, seg_len)
                    if q_idx < query_len:
                        rec.add_cell_event("h_matrix", q_idx + 1, reference_pos, score)

    return best


class FarrarImpl(Aligner):
    """Farrar's striped Smith-Waterman implementation. lanes is set at construction."""

    def __init__(self, lanes: int = 8, verbose: int = 0) -> None:
        self.lanes = lanes
        self.verbose = verbose
        self.rec = Recorder(verbose=verbose)
        self.results: list[AlignmentResult] = []
        self.pair_recs: list[Recorder] = []

    def run(self, pen: array.array) -> None:
        for _qname, qseq, _rname, rseq in self.pairs:
            pair_rec = Recorder(verbose=self.verbose)
            result = _run_pair(qseq, rseq, pen, self.lanes, pair_rec)
            self.results.append(result)
            self.pair_recs.append(pair_rec)
            self.rec.add_time("smith_waterman.dp_fill", pair_rec.times.get("smith_waterman.dp_fill", 0.0))

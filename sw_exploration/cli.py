"""Command-line interface.

All input sources produce a flat list of (query_name, query_seq, ref_name, ref_seq)
pairs. build_pairs() collects them; the chosen implementation then iterates over
all pairs internally via its run() method.

Visualisation (--preview / --heatmap) shows the H matrix per pair, with
optional overlays for lazy-F trigger cells and min-score thresholds.
"""

from __future__ import annotations

import argparse
import array
import random
from collections import Counter

from .fasta import load_fasta_pairs, normalize_sequence
from .output import (
    OVERLAY_REGISTRY,
    build_matrix_figure,
    build_summary_figure,
    farrar_time,
    format_matrix,
    matrix_from_cell_events,
    next_output_path,
    smith_waterman_time,
    write_output,
)
from .sw_wrapper import SCORING_REGISTRY, create_impl
from .sw_implementations.scalar import ScalarImpl


def random_sequence(length: int, rng: random.Random) -> str:
    return "".join(rng.choice("ACGTN") for _ in range(length))


def parse_penalties(s: str) -> array.array:
    """Parse MATCH,MISMATCH,DEL_OPEN,DEL_EXT,INS_OPEN,INS_EXT into an int8 array."""
    parts = s.split(",")
    if len(parts) != 6:
        raise SystemExit(
            "--penalties requires exactly 6 comma-separated integers: "
            "MATCH,MISMATCH,DEL_OPEN,DEL_EXT,INS_OPEN,INS_EXT"
        )
    try:
        values = [int(p) for p in parts]
    except ValueError:
        raise SystemExit("--penalties: all six values must be integers")
    try:
        return array.array('b', values)
    except OverflowError:
        raise SystemExit("--penalties: all values must fit in int8 (-128..127)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Explore Smith-Waterman DP and Farrar's striped method.\n\n"
            "Input sources are additive: combine --single-query/reference, "
            "--query/reference-file, and --random-*-count freely. All sources "
            "produce a flat list of pairs that are aligned in sequence."
        )
    )

    # --- input sources ---
    parser.add_argument("--single-query", dest="single_query", default=None,
                        help="inline query sequence")
    parser.add_argument("--single-reference", "--single-target",
                        dest="single_reference", default=None,
                        help="inline reference sequence")
    parser.add_argument("--query-file", default=None,
                        help="FASTA file of query sequences")
    parser.add_argument("--reference-file", "--target-file",
                        dest="reference_file", default=None,
                        help="FASTA file of reference sequences (paired by index)")
    parser.add_argument("--random-count", type=int, default=None, metavar="N",
                        help="generate N random query/reference pairs")
    parser.add_argument("--query-length", type=int, default=32,
                        help="length of each randomly generated query (default: 32)")
    parser.add_argument("--reference-length", type=int, default=32,
                        help="length of each randomly generated reference (default: 32)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--reverse", action="store_true", help="Reverse sequences")
    parser.add_argument("--transpose", action="store_true",
                        help="swap query and reference for every pair")

    # --- scoring ---
    parser.add_argument(
        "--penalties", default="-2,1,3,1,3,1",
        metavar="M,X,DO,DE,IO,IE",
        help=(
            "scoring penalties as six comma-separated int8 values: "
            "MATCH,MISMATCH,DEL_OPEN,DEL_EXT,INS_OPEN,INS_EXT. "
            "MATCH/MISMATCH are pre-negated: actual score delta is -MATCH on "
            "a match, -MISMATCH on a mismatch "
            "(default: -2,1,3,1,3,1 = +2 reward / -1 penalty)"
        ),
    )
    parser.add_argument(
        "--implementation", default="farrar",
        choices=sorted(SCORING_REGISTRY),
        help="scoring implementation (default: farrar)",
    )
    parser.add_argument("--lanes", type=int, default=8,
                        help="number of SIMD lanes for farrar/c_farrar (default: 8)")

    # --- output ---
    # TODO: fix the verbose argument to allow exlusivity between choices 0-3
    # 0 summary only, minimal output, 1 events/progress, 2 full counters, 3 debug output flag
    parser.add_argument(
        "--verbose", type=int, choices=[0, 1, 2], default=0,
        help="0: summary only, 1: stage events per pair, 2: full counters",
    )
    parser.add_argument("--show-matrix", action="store_true",
                        help="print the Smith-Waterman H matrix for each pair")
    parser.add_argument("--show-triggers", action="store_true",
                        help="print per-pair counts for every recorded cell-event "
                             "trigger label (e.g. farrar.lazy_f_trigger)")
    parser.add_argument(
        "--heatmap", default=None,
        help="save H matrix figure to file (.png, .pdf, etc.)",
    )
    parser.add_argument("--preview", action="store_true",
                        help="open an interactive matplotlib window showing H matrices")
    parser.add_argument("--min-score", type=int, default=None,
                        help="score threshold; enables score_pass_rate metric "
                             "and is used by the min_score overlay")
    parser.add_argument(
        "--heatmap-overlay", dest="heatmap_overlays", metavar="KEY",
        action="append", default=[],
        choices=sorted(OVERLAY_REGISTRY),
        help="overlay to draw on the matrix figure; may be repeated "
             f"(choices: {', '.join(sorted(OVERLAY_REGISTRY))})",
    )
    parser.add_argument(
        "--output", default="results.json",
        help="output JSON file path template (auto-numbered, default: results.json)",
    )
    parser.add_argument("--annotate-heatmap", action="store_true",
                        help="overlay each cell's score value on the heatmap "
                             "(font auto-scales; save high-dpi for large matrices)")
    parser.add_argument("--no-results", action="store_true",
                        help="skip writing the JSON output file")
    parser.add_argument("--validate-scalar", action="store_true",
                        help="also run scalar DP and flag score mismatches")
    parser.add_argument("--summary", action="store_true",
                        help="show a per-pair bar chart (score + lazy-F) instead of H matrices")
    # TODO: remove this argument and refactor the verbosity flag
    parser.add_argument("--progress", action="store_true")

    return parser.parse_args()


def build_pairs(args: argparse.Namespace) -> list[tuple[str, str, str, str]]:
    """Collect all (query_name, query_seq, ref_name, ref_seq) pairs from all input sources."""
    pairs: list[tuple[str, str, str, str]] = []

    if args.single_query is not None or args.single_reference is not None:
        q = normalize_sequence(args.single_query or "ACACACTA", "query")
        r = normalize_sequence(args.single_reference or "AGCACACA", "reference")
        pairs.append(("query", q, "reference", r))

    if args.query_file is not None or args.reference_file is not None:
        if not args.query_file or not args.reference_file:
            raise SystemExit("--query-file and --reference-file must be used together")
        pairs.extend(load_fasta_pairs(args.query_file, args.reference_file))

    if args.random_count is not None:
        rng = random.Random(args.seed)
        for i in range(args.random_count):
            q = random_sequence(args.query_length, rng)
            r = random_sequence(args.reference_length, rng)
            pairs.append((f"query_{i}", q, f"reference_{i}", r))

    if not pairs:
        pairs.append(("query", "ACACACTA", "reference", "AGCACACA"))

    if args.reverse:
        pairs = [(qn, qs[::-1], rn, rs[::-1]) for qn, qs, rn, rs in pairs]

    if args.transpose:
        pairs = [(rn, rs, qn, qs) for qn, qs, rn, rs in pairs]

    return pairs


def main() -> None:
    args = parse_args()
    pairs = build_pairs(args)
    pen = parse_penalties(args.penalties)

    # Insantiate and run Smith-Waterman-Gotoh implementation
    #   ____ _  _ _ ___ _  _    _ _ _ ____ ___ ____ ____ _  _ ____ _  _    ____ ____ ___ ____ _  _ 
    #   [__  |\/| |  |  |__| __ | | | |__|  |  |___ |__/ |\/| |__| |\ | __ | __ |  |  |  |  | |__| 
    #   ___] |  | |  |  |  |    |_|_| |  |  |  |___ |  \ |  | |  | | \|    |__] |__|  |  |__| |  | 
    # This is the main entrance to the rest of the code
    impl = create_impl(args.implementation, args, pairs)
    impl.run(pen)

    # Code after this line is for output, validation, etc.

    # Optionally run scalar DP for score validation (when chosen impl is not scalar).
    scalar_impl = None
    if args.validate_scalar and args.implementation != "scalar":
        scalar_impl = ScalarImpl(verbose=args.verbose)
        scalar_impl.pairs = pairs
        scalar_impl.run(pen)

    summary_only = args.summary and not args.show_matrix
    need_matrix_display = (
        args.show_matrix
        or (args.preview and not summary_only)
        or (bool(args.heatmap) and not summary_only)
    )

    # Build per-pair output data.
    pairs_data: list[dict] = []
    total_times: dict[str, float] = Counter()
    total_counts: Counter[str] = Counter()

    # iterate through alignment
    for index, ((query_name, query_seq, ref_name, ref_seq), pair_best, pair_recorder) in enumerate(zip(impl.pairs, impl.results, impl.pair_recs)):
        # Note we are also iterating through pair_recs and results with index

        # Give TracebackResult, unfortunately it's not implemented yet?!?! what madness?!?!
        dp_fill_time = pair_recorder.times.get("smith_waterman.dp_fill", 0.0)
        if not dp_fill_time and scalar_impl is not None:
            dp_fill_time = scalar_impl.pair_recs[index].times.get("smith_waterman.dp_fill", 0.0)

        print(
            f"pair {index + 1}/{len(pairs)}: {query_name} x {ref_name}\n"
            f"max_score={pair_best.score}"
            f", dp_fill_time={dp_fill_time:.6f}s"
            ,flush=True
        )

        # Use the chosen impl's h_matrix cell events for display; fall back
        # to scalar's when validating.
        h_matrix: list[list[int]] | None = None
        if need_matrix_display:
            h_events = pair_recorder.cell_events.get("h_matrix")
            if h_events is None and scalar_impl is not None:
                h_events = scalar_impl.pair_recs[index].cell_events.get("h_matrix")
            if h_events is not None:
                h_matrix = matrix_from_cell_events(
                    h_events, len(query_seq) + 1, len(ref_seq) + 1
                )

        lazy_f = pair_recorder.counts.get("farrar.lazy_f_corrections", 0)

        # Every non-"h_matrix" cell-event label is a "trigger" position list
        # (e.g. farrar.lazy_f_trigger); kept generic rather than farrar-specific.
        triggers = {
            label: events
            for label, events in pair_recorder.cell_events.items()
            if label != "h_matrix"
        }
        total_times.update(pair_recorder.times)
        total_counts.update(pair_recorder.counts)

        score_mismatch = False
        if scalar_impl is not None:
            score_mismatch = scalar_impl.results[index].score != pair_best.score

        pairs_data.append(
            {
                "pair_index": index,
                "query_name": query_name,
                "reference_name": ref_name,
                "query_seq": query_seq,
                "reference_seq": ref_seq,
                "query_len": len(query_seq),
                "reference_len": len(ref_seq),
                "score": pair_best.score,
                "end_query": pair_best.end_query,
                "end_reference": pair_best.end_reference,
                "lazy_f_corrections": lazy_f,
                "triggers": triggers,
                "score_mismatch": int(score_mismatch),
                "smith_waterman_time_s": smith_waterman_time(pair_recorder.times),
                "farrar_time_s": farrar_time(pair_recorder.times),
                "h_matrix": h_matrix,
            }
        )

        if args.show_triggers:
            for label, events in triggers.items():
                if events:
                    print(
                        f"  pair {index:>3}: {query_name} x {ref_name}"
                        f"  score={pair_best.score}"
                        f"  {label}={len(events)}"
                    )

        if args.show_matrix and h_matrix is not None:
            print(f"\nH matrix for {query_name} x {ref_name}:")
            print(format_matrix(h_matrix, query_seq, ref_seq))

        if args.verbose >= 1:
            label = "full events" if args.verbose >= 2 else "stage events"
            print(f"\n{label} for {query_name} x {ref_name}:")
            for event in pair_recorder.events:
                print(f"  {event}")
    # End of iterating through results
    # Anything afterward can be a summary or needed to loop through all elements first

    print(
        f"Overall time ",
        f"dp_fill_time={impl.rec.times.get("smith_waterman.dp_fill", 0.0):.6f}s"
        ,flush=True
    )


    # TODO: Add options for what to print besides defaults, 
    # These lines are commented out as they don't really reflect getting the stats wanted from everything
    # as the current --summary flag isn't well implemented

    #
    #total_lazy_f = sum(p["lazy_f_corrections"] for p in pairs_data)
    #total_triggers = sum(
    #    len(events) for p in pairs_data for events in p["triggers"].values()
    #)
    #mismatches = sum(p["score_mismatch"] for p in pairs_data)
    #print(
    #    f"pairs={len(pairs)}"
    #    f", max_score={max_score}"
    #    f", score_time={score_time(total_times):.6f}s"
    #)
    #if not args.no_results:
    #    print(f"output: {output_path}")

    #if args.verbose >= 2:
    #    print_counts("full counts", total_counts)
    #    print_times(total_times)
    #

    if args.preview or args.heatmap:
        import matplotlib.pyplot as plt
        if getattr(args, "summary", False):
            fig = build_summary_figure(pairs_data)
        else:
            vis_pairs = [p for p in pairs_data if p["h_matrix"] is not None]
            if not vis_pairs:
                print("warning: no H matrices available for visualisation")
                vis_pairs = None
            if vis_pairs:
                fig = build_matrix_figure(
                    vis_pairs,
                    overlays=args.heatmap_overlays,
                    annotate=args.annotate_heatmap,
                )
            else:
                fig = None
        if fig is not None:
            if args.heatmap:
                fig.savefig(args.heatmap, bbox_inches="tight")
                print(f"heatmap: {args.heatmap}")
            if args.preview:
                plt.show()
            plt.close(fig)


if __name__ == "__main__":
    main()

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
    next_output_path,
    print_counts,
    print_times,
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
    parser.add_argument(
        "--verbose", type=int, choices=[0, 1, 2], default=0,
        help="0: summary only, 1: stage events per pair, 2: full counters",
    )
    parser.add_argument("--show-matrix", action="store_true",
                        help="print the Smith-Waterman H matrix for each pair")
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
    parser.add_argument("--transpose", action="store_true",
                        help="swap query and reference for every pair")
    parser.add_argument("--validate-scalar", action="store_true",
                        help="also run scalar DP and flag score mismatches")
    parser.add_argument("--summary", action="store_true",
                        help="show a per-pair bar chart (score + lazy-F) instead of H matrices")
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

    if args.transpose:
        pairs = [(rn, rs, qn, qs) for qn, qs, rn, rs in pairs]

    return pairs


def main() -> None:
    args = parse_args()
    pairs = build_pairs(args)
    pen = parse_penalties(args.penalties)

    # Run the chosen implementation over all pairs.
    impl = create_impl(args.implementation, args)
    impl.run(pairs, pen)

    # Optionally run scalar DP for score validation (when chosen impl is not scalar).
    scalar_impl = None
    if args.validate_scalar and args.implementation != "scalar":
        scalar_impl = ScalarImpl(verbose=args.verbose)
        scalar_impl.run(pairs, pen)

    summary_only = args.summary and not args.show_matrix
    need_matrix_display = (
        args.show_matrix
        or (args.preview and not summary_only)
        or (bool(args.heatmap) and not summary_only)
    )

    # Build per-pair output data.
    pairs_data: list[dict] = []
    total_times: Counter[str] = Counter()
    total_counts: Counter[str] = Counter()

    for index, (query_name, query_seq, ref_name, ref_seq) in enumerate(pairs):
        if args.progress:
            print(f"pair {index + 1}/{len(pairs)}: {query_name} x {ref_name}", flush=True)

        result = impl.results[index]
        pair_rec = impl.pair_recs[index]

        # Use the chosen impl's h_matrix for display; fall back to scalar's when validating.
        h_matrix: list[list[int]] | None = None
        if need_matrix_display:
            if hasattr(impl, 'h_matrices') and impl.h_matrices[index] is not None:
                h_matrix = impl.h_matrices[index]
            elif scalar_impl is not None:
                h_matrix = scalar_impl.h_matrices[index]

        lazy_f = pair_rec.counts.get("farrar.lazy_f_corrections", 0)
        lazy_f_triggers = pair_rec.cell_events.get("farrar.lazy_f_trigger", [])
        total_times.update(pair_rec.times)
        total_counts.update(pair_rec.counts)

        score_mismatch = False
        if scalar_impl is not None:
            score_mismatch = scalar_impl.results[index].score != result.score

        pairs_data.append(
            {
                "pair_index": index,
                "query_name": query_name,
                "reference_name": ref_name,
                "query_seq": query_seq,
                "reference_seq": ref_seq,
                "query_len": len(query_seq),
                "reference_len": len(ref_seq),
                "score": result.score,
                "end_query": result.end_query,
                "end_reference": result.end_reference,
                "lazy_f_corrections": lazy_f,
                "lazy_f_triggers": lazy_f_triggers,
                "score_mismatch": int(score_mismatch),
                "smith_waterman_time_s": smith_waterman_time(pair_rec.times),
                "farrar_time_s": farrar_time(pair_rec.times),
                "h_matrix": h_matrix,
            }
        )

        if lazy_f > 0:
            print(
                f"  pair {index:>3}: {query_name} x {ref_name}"
                f"  score={result.score}"
                f"  lazy_f_corrections={lazy_f}"
                f"  lazy_f_triggers={len(lazy_f_triggers)}"
            )

        if args.show_matrix and h_matrix is not None:
            print(f"\nH matrix for {query_name} x {ref_name}:")
            print(format_matrix(h_matrix, query_seq, ref_seq))

        if args.verbose >= 1:
            label = "full events" if args.verbose >= 2 else "stage events"
            print(f"\n{label} for {query_name} x {ref_name}:")
            for event in pair_rec.events:
                print(f"  {event}")

    if not args.no_results:
        output_path = next_output_path(args.output)
        write_output(output_path, pairs_data, pen, args)

    total_lazy_f = sum(p["lazy_f_corrections"] for p in pairs_data)
    total_triggers = sum(len(p["lazy_f_triggers"]) for p in pairs_data)
    mismatches = sum(p["score_mismatch"] for p in pairs_data)
    print(
        f"pairs={len(pairs)}"
        f", lazy_f_corrections={total_lazy_f}"
        f", lazy_f_trigger_cells={total_triggers}"
        f", mismatches={mismatches}"
        f", farrar_time={farrar_time(total_times):.6f}s"
    )
    if not args.no_results:
        print(f"output: {output_path}")

    if args.verbose >= 2:
        print_counts("full counts", total_counts)
        print_times(total_times)

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

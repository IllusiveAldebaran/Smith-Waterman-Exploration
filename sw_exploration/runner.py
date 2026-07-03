"""run_one_pair: execute scalar DP and/or the chosen implementation on one pair.

Shared inner loop for all processing paths. validate_scalar and the presence
of show_matrix/preview/heatmap flags control whether the scalar H matrix is
computed and returned. cell_events from the Recorder carry per-cell data such
as lazy-F trigger coordinates.
"""

from __future__ import annotations

import argparse
from collections import Counter

from .sw_wrapper import run_scoring, smith_waterman_dp, traceback_alignment
from .types import AlignmentResult, Recorder


def run_one_pair(
    query: str,
    reference: str,
    args: argparse.Namespace,
    validate_scalar: bool,
) -> tuple[
    AlignmentResult,
    bool,
    Counter[str],
    Counter[str],
    list[str],
    list[list[int]] | None,
    dict[str, list[tuple[int, int]]],
]:
    """Align one query/reference pair and return instrumentation data.

    Returns (result, score_mismatch, counts, times, events, matrix, cell_events).

    matrix is the scalar H matrix when the args request visualisation or
    show_matrix; otherwise None. cell_events carries sparse per-cell data
    recorded during the run (e.g. farrar.lazy_f_trigger coordinates).
    """
    rec = Recorder(verbose=args.verbose)
    score_mismatch = False
    scalar_score: int | None = None
    matrix: list[list[int]] | None = None

    summary_only = getattr(args, "summary", False) and not getattr(args, "show_matrix", False)
    need_matrix = (
        validate_scalar
        or getattr(args, "show_matrix", False)
        or (getattr(args, "preview", False) and not summary_only)
        or (bool(getattr(args, "heatmap", None)) and not summary_only)
    )

    if need_matrix:
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

    result = run_scoring(
        args.implementation,
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
        score_mismatch = scalar_score != result.score

    return result, score_mismatch, rec.counts, rec.times, rec.events, matrix, rec.cell_events

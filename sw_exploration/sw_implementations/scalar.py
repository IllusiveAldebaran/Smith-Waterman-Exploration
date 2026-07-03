"""Scalar affine-gap Smith-Waterman local alignment.

Fills three DP matrices (H, E, F) and a pointer matrix for traceback.
This is the reference implementation: correct, readable, and slow. Every
major step is counted and timed via a Recorder so callers can compare
operation counts against faster implementations.

Canonical scoring signature (used by sw_wrapper.SCORING_REGISTRY):
  run(query, reference, match, mismatch, gap_open, gap_extend, lanes, rec)
      -> AlignmentResult
The `lanes` parameter is accepted but unused; it exists so all registered
implementations share the same call signature.
"""

from __future__ import annotations

from ..types import AlignmentResult, Recorder, TracebackResult, NEG_INF


def score_pair(a: str, b: str, match: int, mismatch: int) -> int:
    """Return match score if residues are identical, mismatch score otherwise."""
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

    Returns (best_result, H_matrix, ptr_matrix) so the caller can run
    traceback_alignment if the full alignment path is needed.
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


def run(
    query: str,
    reference: str,
    match: int,
    mismatch: int,
    gap_open: int,
    gap_extend: int,
    lanes: int,  # unused; present for uniform signature across implementations
    rec: Recorder,
) -> AlignmentResult:
    """Canonical scoring entry point for SCORING_REGISTRY dispatch.

    Runs the full DP but discards the matrix; use smith_waterman_dp directly
    when you also need the traceback matrices.
    """
    result, _, _ = smith_waterman_dp(
        query, reference, match, mismatch, gap_open, gap_extend, rec
    )
    return result

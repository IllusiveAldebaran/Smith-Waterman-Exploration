"""Scalar affine-gap Smith-Waterman local alignment.

Fills three DP matrices (H, E, F) and a pointer matrix for traceback.
This is the reference implementation: correct, readable, and slow. Every
major step is counted and timed via a Recorder so callers can compare
operation counts against faster implementations.

pen layout: array.array('b', [match, mismatch, del_open, del_ext, ins_open, ins_ext])
  del_open/del_ext apply to E (horizontal gaps).
  ins_open/ins_ext apply to F (vertical gaps).
  match/mismatch are pre-negated: the actual score contribution is -match on
  a match and -mismatch on a mismatch (default -2,1 for a +2 reward / -1 penalty).
"""

from __future__ import annotations

import array

from ..types import Aligner, AlignmentResult, Recorder, TracebackResult, NEG_INF


def score_pair(a: str, b: str, match: int, mismatch: int) -> int:
    """Return match/mismatch score. match/mismatch are pre-negated; negate back here."""
    return -match if a == b else -mismatch


def smith_waterman_dp(
    query: str,
    reference: str,
    pen: array.array,
    rec: Recorder,
) -> tuple[AlignmentResult, list[list[int]], list[list[str]]]:
    """Fill the scalar Smith-Waterman affine-gap DP matrices.

    H is the best local score ending at each cell.
    E is the best score ending with a horizontal gap (del_open/del_ext).
    F is the best score ending with a vertical gap (ins_open/ins_ext).
    ptr records the selected predecessor for traceback.

    Returns (best_result, H_matrix, ptr_matrix).
    """
    match, mismatch, del_open, del_ext, ins_open, ins_ext = pen
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
                e[i][j] = max(h[i][j - 1] - del_open, e[i][j - 1] - del_ext)

                rec.count("smith_waterman.gap_f_updates")
                f[i][j] = max(h[i - 1][j] - ins_open, f[i - 1][j] - ins_ext)

                rec.count("smith_waterman.diagonal_updates")
                diag = h[i - 1][j - 1] + score_pair(
                    query[i - 1], reference[j - 1], match, mismatch
                )

                rec.count("smith_waterman.cell_max_reductions")
                h[i][j] = max(0, diag, e[i][j], f[i][j])
                rec.add_cell_event("h_matrix", i, j, h[i][j])

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


class ScalarImpl(Aligner):
    """Reference scalar affine-gap Smith-Waterman implementation."""

    def __init__(self, verbose: int = 0) -> None:
        self.verbose = verbose
        self.rec = Recorder(verbose=verbose)
        self.results: list[AlignmentResult] = []
        self.pair_recs: list[Recorder] = []

    def run(self, pen: array.array) -> None:
        # Separate direction matrix also stored in ptr for traceback
        for _qname, qseq, _rname, rseq in self.pairs:
            pair_rec = Recorder(verbose=self.verbose)
            result, h, ptr = smith_waterman_dp(qseq, rseq, pen, pair_rec)
            traceback_alignment(qseq, rseq, h, ptr, result, pair_rec)
            self.results.append(result)
            self.pair_recs.append(pair_rec)
            self.rec.add_time("smith_waterman.dp_fill", pair_rec.times.get("smith_waterman.dp_fill", 0.0))

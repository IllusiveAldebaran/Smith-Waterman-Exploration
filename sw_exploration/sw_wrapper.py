"""Runtime dispatch wrapper for Smith-Waterman implementations.

All concrete implementations live in sw_implementations/ and expose only
run_batch().  This module imports them, registers them in SCORING_REGISTRY,
and exposes run_scoring() for name-based single-pair dispatch (used by
runner.py).  It also re-exports the scalar DP helpers needed for matrix
display and traceback.

Adding a new implementation
---------------------------
1. Create sw_implementations/myimpl.py with:

     def run_batch(
         max_query_len, max_reference_len,
         match, mismatch, gap_open, gap_extend,
         queries, references,
         lanes=8, rec=None,
     ) -> tuple[list[AlignmentResult], list[list[list[int]] | None]]:
         ...

2. Register it here:

     from .sw_implementations import myimpl
     SCORING_REGISTRY["myimpl"] = myimpl.run_batch

That is all. The CLI --implementation flag selects from SCORING_REGISTRY at
runtime.
"""

from __future__ import annotations

from .sw_implementations import c_farrar, c_scalar, farrar, scalar
from .sw_implementations.scalar import smith_waterman_dp, traceback_alignment
from .types import AlignmentResult, Recorder

# ---------------------------------------------------------------------------
# Registry — maps implementation name to its run_batch function.
# Each entry must accept:
#   (max_query_len, max_ref_len, match, mismatch, gap_open, gap_extend,
#    queries, references, lanes=8, rec=None)
# and return (list[AlignmentResult], list[h_matrix | None]).
# ---------------------------------------------------------------------------
SCORING_REGISTRY: dict[str, object] = {
    "scalar":   scalar.run_batch,
    "farrar":   farrar.run_batch,
    "c_scalar": c_scalar.run_batch,
    "c_farrar": c_farrar.run_batch,
}


def run_scoring(
    name: str,
    query: str,
    reference: str,
    match: int,
    mismatch: int,
    gap_open: int,
    gap_extend: int,
    lanes: int,
    rec: Recorder,
) -> AlignmentResult:
    """Dispatch a single pair to a named implementation via run_batch.

    Packages the pair as a batch of 1 and returns results[0].  The H matrix
    from run_batch is discarded here; the display matrix comes from
    smith_waterman_dp in runner.py when need_matrix is True.

    Raises ValueError for unknown names.
    """
    impl = SCORING_REGISTRY.get(name)
    if impl is None:
        available = ", ".join(sorted(SCORING_REGISTRY))
        raise ValueError(
            f"unknown implementation {name!r}; available: {available}"
        )
    results, _ = impl(
        len(query), len(reference),
        match, mismatch, gap_open, gap_extend,
        query, reference,
        lanes=lanes, rec=rec,
    )
    return results[0]

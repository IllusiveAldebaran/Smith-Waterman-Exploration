"""Runtime dispatch wrapper for Smith-Waterman implementations.

All concrete implementations live in sw_implementations/. This module imports
them, registers them in SCORING_REGISTRY, and exposes run_scoring() for
name-based dispatch. It also re-exports the scalar DP functions that need the
full matrix (for traceback and single-pair mode).

Adding a new implementation
---------------------------
1. Create sw_implementations/myimpl.py with a `run` function:

     def run(query, reference, match, mismatch, gap_open, gap_extend, lanes, rec)
         -> AlignmentResult

   The `lanes` parameter is available for striped/SIMD implementations;
   purely scalar ones may ignore it. Use the Recorder for all counts/timings.

2. Register it here:

     from .sw_implementations import myimpl
     SCORING_REGISTRY["myimpl"] = myimpl.run

That is all. The CLI flag --implementation selects from SCORING_REGISTRY at
runtime, and batch/grid mode will use whichever entry is chosen.
"""

from __future__ import annotations

from .sw_implementations import farrar, scalar
from .sw_implementations.scalar import smith_waterman_dp, traceback_alignment
from .types import AlignmentResult, Recorder

# ---------------------------------------------------------------------------
# Registry — maps implementation name to its canonical scoring function.
# Each entry must accept:
#   (query, reference, match, mismatch, gap_open, gap_extend, lanes, rec)
# and return an AlignmentResult.
# ---------------------------------------------------------------------------
SCORING_REGISTRY: dict[str, object] = {
    "scalar": scalar.run,
    "farrar": farrar.run,
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
    """Dispatch to a named implementation from SCORING_REGISTRY.

    Raises ValueError for unknown names so the error message lists what is
    actually available rather than giving a raw KeyError.
    """
    impl = SCORING_REGISTRY.get(name)
    if impl is None:
        available = ", ".join(sorted(SCORING_REGISTRY))
        raise ValueError(
            f"unknown implementation {name!r}; available: {available}"
        )
    return impl(query, reference, match, mismatch, gap_open, gap_extend, lanes, rec)

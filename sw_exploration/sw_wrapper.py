"""Runtime dispatch wrapper for Smith-Waterman-Gotoh implementations.

All concrete implementations live in sw_implementations/ and expose a class
that subclasses Aligner.  This module imports them, registers them in
SCORING_REGISTRY, and exposes create_impl() for CLI instantiation.

Adding a new implementation
---------------------------
1. Create sw_implementations/myimpl.py with a class MyImpl(Aligner).
2. Register it here:

     from .sw_implementations import myimpl
     SCORING_REGISTRY["myimpl"] = myimpl.MyImpl

That is all. The CLI --implementation flag selects from SCORING_REGISTRY at runtime.
"""

from __future__ import annotations

import argparse

from .sw_implementations import c_farrar, c_scalar, farrar, scalar, hip_diagonal
from .types import Aligner

# ---------------------------------------------------------------------------
# Registry — maps implementation name to its class.
# Each class must subclass Aligner and accept at minimum verbose=0 in its
# constructor. Implementation-specific params (e.g. lanes) are also accepted
# by the classes that need them.
# ---------------------------------------------------------------------------
SCORING_REGISTRY: dict[str, type[Aligner]] = {
    "scalar":   scalar.ScalarImpl,
    "farrar":   farrar.FarrarImpl,
    "c_scalar": c_scalar.CScalarImpl,
    "c_farrar": c_farrar.CFarrarImpl,
    "hip_diagonal": hip_diagonal.HIPDiagonalImpl,
}

# Implementations that accept a lanes parameter at construction.
_LANES_IMPLS = {"farrar", "hip_diagonal"}

# Implementations that accept a second_pass parameter at construction.
_SECOND_PASS_IMPLS = {"hip_diagonal"}
_BEST_SCORE_IMPLS = {"best_score"}

# ---------------------------------------------------------------------------
# Per-implementation capability facts for --implementation-options (cli.py).
# `lanes`/`second_pass`/`best_cell_only`/`best_score` are derived from the membership sets
# and registry above so they can't drift from what create_impl() actually
# does; `full_traceback`/`gpu`/notes are facts the sets above don't capture,
# so they're kept here as plain data. When adding a new implementation, add
# an entry to _FULL_TRACEBACK / _GPU / _NOTES too -- missing ones fall back
# to "unknown" rather than breaking --implementation-options.
# ---------------------------------------------------------------------------
_FULL_TRACEBACK: dict[str, object] = {
    "scalar": True,
    "farrar": False,
    "c_scalar": False,
    "c_farrar": "yes, unless best_cell_only",
    "hip_diagonal": "second_pass only",
}
_GPU: dict[str, bool] = {
    "scalar": False,
    "farrar": False,
    "c_scalar": False,
    "c_farrar": False,
    "hip_diagonal": True,
}
_NOTES: dict[str, str] = {
    "scalar": (
        "Reference pure-Python affine-gap DP. best_cell_only only skips the "
        "traceback_alignment() call and reuses the AlignmentResult "
        "smith_waterman_dp() already returns -- the full H/E/F/ptr matrices "
        "are still filled either way, since a hand-rolled reduced-memory "
        "variant wouldn't be meaningfully faster in pure Python. "
        "traceback_alignment()'s result isn't wired into CLI output yet."
    ),
    "farrar": (
        "Farrar's striped SIMD method simulated in Python. best_cell_only is "
        "a no-op -- always records h_matrix/farrar.lazy_f_trigger "
        "cell_events. Never builds a ptr matrix, so no traceback is "
        "possible regardless."
    ),
    "c_scalar": (
        "C-backed scalar DP via cffi. best_cell_only skips copying H_buf "
        "into per-cell Recorder events, not the C-side fill itself. No ptr "
        "matrix is returned to Python, so no traceback path exists yet."
    ),
    "c_farrar": (
        "C-backed Farrar's striped method via cffi, using the vendored SSW "
        "library (SSE2) as the kernel instead of simulating it in Python. "
        "best_cell_only=False (the default) gets a real traceback for free "
        "-- SSW's own reverse pass + banded-DP cigar pass, decoded into a "
        "TracebackResult here. Only supports one affine gap cost applied to "
        "both insertions and deletions; run() raises SystemExit up front "
        "if --penalties asks for asymmetric del/ins costs rather than "
        "aligning with the wrong cost."
    ),
    "hip_diagonal": (
        "HIP/GPU diagonal-striped DP. H/E/F live and die on the device -- "
        "best_cell_only and best_score is a no-op there (nothing is ever "
        "copied back regardless). --second-pass gets a real traceback via a "
        "scalar CPU fallback per pair selected by _needs_backtrace() (currently "
        "a stub that always returns True); there's no GPU traceback kernel yet."
    ),
}


def _build_implementation_options() -> dict[str, dict[str, object]]:
    return {
        name: {
            "lanes": name in _LANES_IMPLS,
            "best_cell_only": True,
            "best_score": True,
            "second_pass": name in _SECOND_PASS_IMPLS,
            "best_score": name in _BEST_SCORE_IMPLS,
            "full_traceback": _FULL_TRACEBACK.get(name, "unknown"),
            "gpu": _GPU.get(name, "unknown"),
            "notes": _NOTES.get(name, ""),
        }
        for name in SCORING_REGISTRY
    }


# Keyed by implementation name; see cli.py's --implementation-options.
IMPLEMENTATION_OPTIONS: dict[str, dict[str, object]] = _build_implementation_options()


def create_impl(
    name: str,
    args: argparse.Namespace,
    pairs: list[tuple[str, str, str, str]],
) -> Aligner:
    """Instantiate a named implementation with relevant args and its pairs.

    Passes verbose to all implementations. Passes lanes to implementations
    that accept it (farrar, c_farrar). Sets .pairs on the instance so run()
    can iterate over them.

    Raises ValueError for unknown names.
    """
    cls = SCORING_REGISTRY.get(name)
    if cls is None:
        available = ", ".join(sorted(SCORING_REGISTRY))
        raise ValueError(f"unknown implementation {name!r}; available: {available}")
    kwargs: dict = {
        "verbose": getattr(args, "verbose", 0),
        "best_cell_only": getattr(args, "best_cell_only", False),
    }
    # Haven't implemented different lanes right now (SSE, AVX2, AVX512)
    if name in _BEST_SCORE_IMPLS:
        kwargs["best_score"] = getattr(args, "best_score", False)
    if name in _LANES_IMPLS and args.lanes is not None:
        kwargs["lanes"] = args.lanes
    if name in _SECOND_PASS_IMPLS:
        kwargs["second_pass"] = getattr(args, "second_pass", False)
    impl = cls(**kwargs)
    impl.pairs = pairs
    return impl

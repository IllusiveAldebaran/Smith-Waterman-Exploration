"""Runtime dispatch wrapper for Smith-Waterman implementations.

All concrete implementations live in sw_implementations/ and expose a class
that subclasses AlgorithmImplementation.  This module imports them, registers
them in SCORING_REGISTRY, and exposes create_impl() for CLI instantiation.

Adding a new implementation
---------------------------
1. Create sw_implementations/myimpl.py with a class MyImpl(AlgorithmImplementation).
2. Register it here:

     from .sw_implementations import myimpl
     SCORING_REGISTRY["myimpl"] = myimpl.MyImpl

That is all. The CLI --implementation flag selects from SCORING_REGISTRY at runtime.
"""

from __future__ import annotations

import argparse

from .sw_implementations import c_farrar, c_scalar, farrar, scalar
from .types import AlgorithmImplementation

# ---------------------------------------------------------------------------
# Registry — maps implementation name to its class.
# Each class must subclass AlgorithmImplementation and accept at minimum
# verbose=0 in its constructor. Implementation-specific params (e.g. lanes)
# are also accepted by the classes that need them.
# ---------------------------------------------------------------------------
SCORING_REGISTRY: dict[str, type[AlgorithmImplementation]] = {
    "scalar":   scalar.ScalarImpl,
    "farrar":   farrar.FarrarImpl,
    "c_scalar": c_scalar.CScalarImpl,
    "c_farrar": c_farrar.CFarrarImpl,
}

# Implementations that accept a lanes parameter at construction.
_LANES_IMPLS = {"farrar", "c_farrar"}


def create_impl(name: str, args: argparse.Namespace) -> AlgorithmImplementation:
    """Instantiate a named implementation with relevant args.

    Passes verbose to all implementations. Passes lanes to implementations
    that accept it (farrar, c_farrar).

    Raises ValueError for unknown names.
    """
    cls = SCORING_REGISTRY.get(name)
    if cls is None:
        available = ", ".join(sorted(SCORING_REGISTRY))
        raise ValueError(f"unknown implementation {name!r}; available: {available}")
    kwargs: dict = {"verbose": getattr(args, "verbose", 0)}
    if name in _LANES_IMPLS:
        kwargs["lanes"] = getattr(args, "lanes", 8)
    return cls(**kwargs)

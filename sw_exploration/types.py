"""Shared data structures and the Recorder instrumentation helper.

AlignmentResult and TracebackResult are the return types of the DP and
traceback passes. Recorder collects named occurrence counts, stage timings,
and per-cell events so callers can answer questions like "how many DP cells
were filled?", "how much time was spent in the lazy-F correction?", and
"which alignment cells triggered lazy-F propagation?".
"""

from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass
from time import perf_counter

NEG_INF = -(10**9)


@dataclass(frozen=True)
class AlignmentResult:
    score: int
    end_query: int
    end_reference: int


@dataclass(frozen=True)
class TracebackResult:
    start_query: int
    start_reference: int
    aligned_query: str
    aligned_reference: str
    path: tuple[str, ...]


class Recorder:
    """Collect occurrence counts, stage timings, and per-cell events.

    Counters are named after algorithmic events rather than low-level Python
    operations. cell_events stores sparse (row, col) coordinates for named
    per-cell events such as lazy-F trigger positions in the H matrix.
    """

    def __init__(self, verbose: int = 0) -> None:
        self.counts: Counter[str] = Counter()
        self.times: Counter[str] = Counter()
        self.verbose = verbose
        self.events: list[str] = []
        self.cell_events: dict[str, list[tuple[int, int]]] = {}

    def count(self, name: str, amount: int = 1) -> None:
        self.counts[name] += amount
        if self.verbose >= 1:
            self.events.append(f"count {name} += {amount}")

    def add_cell_event(self, name: str, row: int, col: int) -> None:
        """Record a (row, col) position in H-matrix coordinates for a named event."""
        if name not in self.cell_events:
            self.cell_events[name] = []
        self.cell_events[name].append((row, col))

    @contextmanager
    def timed(self, name: str):
        if self.verbose >= 1:
            self.events.append(f"start {name}")
        start = perf_counter()
        try:
            yield
        finally:
            elapsed = perf_counter() - start
            self.times[name] += elapsed
            if self.verbose >= 1:
                self.events.append(f"end {name}: {elapsed:.9f}s")

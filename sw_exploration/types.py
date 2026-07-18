"""Shared data structures and the Recorder instrumentation helper.

AlignmentResult and TracebackResult are the return types of the DP and
traceback passes. Recorder collects named occurrence counts, stage timings,
and per-cell events so callers can answer questions like "how many DP cells
were filled?", "how much time was spent in the lazy-F correction?", "which
alignment cells triggered lazy-F propagation?", and "what score did every
cell in the H matrix end up with?".

Aligner is the abstract base class for all scoring implementations. Each
concrete class iterates over self.pairs in run(), storing one AlignmentResult
per pair in self.results and one per-pair Recorder in self.pair_recs.
"""

from __future__ import annotations

import array
from abc import ABC, abstractmethod
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass
from time import perf_counter

NEG_INF = -(10**9)


# Marks the location of the best score in grid
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
    operations. cell_events stores (row, col, value) entries in H-matrix
    coordinates for named per-cell events. value is None for events that
    only mark a position (e.g. a lazy-F trigger); it carries the cell's
    score for events that record a full matrix, one entry per cell (e.g.
    "h_matrix").
    """

    def __init__(self, verbose: int = 0) -> None:
        self.counts: Counter[str] = Counter()
        self.times: dict[str, float] = Counter()
        self.verbose = verbose
        self.events: list[str] = []
        self.cell_events: dict[str, list[tuple[int, int, int | None]]] = {}

    def count(self, name: str, amount: int = 1) -> None:
        self.counts[name] += amount
        if self.verbose >= 1:
            self.events.append(f"count {name} += {amount}")

    def add_time(self, name: str, seconds: float) -> None:
        """Add a precomputed duration (e.g. summed from a nested Recorder) to times."""
        self.times[name] += seconds
        if self.verbose >= 1:
            self.events.append(f"add_time {name} += {seconds:.9f}s")

    def add_cell_event(
        self, name: str, row: int, col: int, value: int | None = None
    ) -> None:
        """Record a (row, col, value) entry in H-matrix coordinates for a named event."""
        if name not in self.cell_events:
            self.cell_events[name] = []
        self.cell_events[name].append((row, col, value))

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


class Aligner(ABC):
    """Abstract base for all Smith-Waterman implementations.

    Subclasses iterate over self.pairs in run(), aligning each pair with its
    own per-pair Recorder. After run() returns:
      self.results[i]   — AlignmentResult for pairs[i]
      self.pair_recs[i]  — per-pair Recorder for pairs[i], including any
                            per-cell events (e.g. a full "h_matrix" event
                            per cell) that implementation chooses to record

    self.pairs is populated by the caller (see sw_wrapper.create_impl)
    before run() is invoked.
    """

    rec: Recorder
    pairs: list[tuple[str, str, str, str]]
    results: list[AlignmentResult]
    pair_recs: list[Recorder]

    @abstractmethod
    def __init__(self, verbose: int = 0) -> None: ...

    @abstractmethod
    def run(self, pen: array.array) -> None:
        """Align every pair in self.pairs and store one result per pair.

        pen: array.array('b', [match, mismatch, del_open, del_ext, ins_open, ins_ext])
        """

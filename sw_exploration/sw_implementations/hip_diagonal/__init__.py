"""C-backed scalar Smith-Waterman via cffi (compiled from scalar.c / swag.h).

pen layout: array.array('b', [match, mismatch, del_open, del_ext, ins_open, ins_ext])
match/mismatch are pre-negated; all six values are passed directly through
as the penalties[6] array.
"""

from __future__ import annotations

import array
import os

from cffi import FFI

from ..scalar import smith_waterman_dp, traceback_alignment
from ...types import Aligner, AlignmentResult, Recorder, TracebackResult

_here = os.path.dirname(os.path.abspath(__file__))


def _build_lib():
    """
    Built hip kernel from diagonal.so built form diagonal.c
    """
    ffi = FFI()

    # cdef() is a pure C declaration parser — it cannot handle preprocessor
    # directives.  Read swag.h but strip any line starting with '#' before
    # passing; the declarations themselves parse fine.
    _h_path = os.path.join(_here, "diagonal.h")
    with open(_h_path) as f:
        _header_decls = "\n".join(
            line for line in f if not line.lstrip().startswith("#") 
                and not line.lstrip().startswith("extern")
                and not line.lstrip().startswith("extern")
                and line.strip() != "{"
                and line.strip() != "}"
        )
    ffi.cdef(_header_decls)

    _DEFAULT_SO_PATH = os.path.join(_here, "libhipdiagonal.so")
    so_path = os.environ.get("SWAG_DIAGONAL_HIP_LIB", _DEFAULT_SO_PATH)

    try:
        lib = ffi.dlopen(so_path)
    except OSError as exc:
        raise ImportError(f"hip_diagonal: failed to load {so_path}") from exc
    return lib, ffi


class HIPDiagonalImpl(Aligner):
    """HIP GPU accelerated Smith-Waterman implementation."""

    def __init__(
        self,
        lanes: int = 8,
        verbose: int = 0,
        best_cell_only: bool = False,
        second_pass: bool = False,
    ) -> None:
        self.verbose = verbose
        self.lanes = lanes
        # H/E/F never leave the device for this implementation regardless (see
        # _align_batch's warning below), so best_cell_only is a no-op here --
        # accepted only so the flag can be passed uniformly across impls.
        self.best_cell_only = best_cell_only
        self.second_pass = second_pass
        self.rec = Recorder(verbose=verbose)
        self.results: list[AlignmentResult] = []
        self.pair_recs: list[Recorder] = []
        # One entry per pair once run() returns, mirroring results/pair_recs.
        # None unless second_pass is enabled and _needs_backtrace() selected
        # that pair -- see _second_pass_backtrace().
        self.tracebacks: list[TracebackResult | None] = []
        self._lib, self._ffi = _build_lib()

    def _align_batch(self, pen: array.array) -> None:
        """Align a batch of reference and query pairs via the C kernel.
        Packs all of the sequences together before passing into C kernel.
        Best score of each pair is stored.

        Records/Copies back into recorders
        WARNING: H/E/F DP matrices live GPU-side only (see diagonal.c) and are
        never copied back to the host -- only best_cell (score, end_query,
        end_reference) survives per pair. h_matrix cell events are not
        recorded for this implementation (--show-matrix/--heatmap/--preview
        won't have data to show for hip_diagonal); materializing the whole
        batch's H matrix on the host was the actual cause of a MemoryError at
        large numAligns.
        WARNING: Post score calculation traceback can be done.. but it's not
        WARNING: Assumes lengths of all queries are the same
        WARNING: Assumes lengths of all references are the same
        """
        # Getting elements from class attributes
        num_pairs = len(self.pairs)

        # This can be edited by dev for profiling
        N_FLOAT_COUNTERS = 0; # for example 1 or num_pairs and timing every pair in C code
        N_INT_COUNTERS = 0
        # Useful for storing info about matrices like timing and counters
        float_counters = self._ffi.new("float[]", N_FLOAT_COUNTERS) if N_FLOAT_COUNTERS > 0 else self._ffi.NULL
        int_counters = self._ffi.new("int[]", N_INT_COUNTERS) if N_INT_COUNTERS > 0 else self._ffi.NULL


        # padded lenngths, takes first item assumes all are the same length
        max_reflen = len(self.pairs[0][3])
        max_qrylen = len(self.pairs[0][1])
        ref_bytes = b''.join((b'\x00' + str(pair[3]).encode('ascii')).ljust(max_reflen+1, b'\x00') for pair in self.pairs)
        qry_bytes = b''.join((b'\x00' + str(pair[1]).encode('ascii')).ljust(max_qrylen+1, b'\x00') for pair in self.pairs)

        # index by 0 to make it clear we're passing the value, not the pointer.
        penalties = self._ffi.new("const struct Penalties*", list(pen))[0]
        best_cell = self._ffi.new("struct bestCell[]", num_pairs)

        with self.rec.timed("smith_waterman.dp_fill"):
            self._lib.alignBatchNpar(
                max_reflen, max_qrylen, penalties, ref_bytes, qry_bytes,
                best_cell,
                num_pairs,
                self.lanes, # lanes is going to be the number of threads called in the GPU
                float_counters, N_FLOAT_COUNTERS,
                int_counters, N_INT_COUNTERS
            )

        for np in range(num_pairs):
            # no longer records the H outputs... too many for one batch!
            #pair_rec = Recorder()
            #h_offset = np * (max_reflen+1) * (max_qrylen+1)
            #res_offset =  (max_reflen+1) * (max_qrylen+1)
            #for i in range(max_reflen+1):
            #    for j in range(max_qrylen+1):
            #        pair_rec.add_cell_event("h_matrix", j, i, H_buf[h_offset+j*(max_reflen+1)+i])
            self.pair_recs.append(Recorder())
            self.results.append(AlignmentResult(best_cell[np].score, best_cell[np].row, best_cell[np].col))


    # FUNCTION DEPRECATED IN FAVOR OF BATCHING
    # Since C implementations are meant to run faster the preferred method is to run in a batch.
    def _align_one(self, pair_index: int, pen: array.array) -> None:
        """Align one pair via the C kernel.

        Records one "h_matrix" cell event per filled cell into rec.
        """
        # Getting elements from class attributes
        qname, qseq, rname, rseq = self.pairs[pair_index]

        # This can be edited by dev for profiling
        N_FLOAT_COUNTERS = 0; # for example 1 or num_pairs and timing every pair in C code
        N_INT_COUNTERS = 0
        # Useful for storing info about matrices like timing and counters
        float_counters = self._ffi.new("float[]", N_FLOAT_COUNTERS) if N_FLOAT_COUNTERS > 0 else self._ffi.NULL
        int_counters = self._ffi.new("int[]", N_INT_COUNTERS) if N_INT_COUNTERS > 0 else self._ffi.NULL

        pair_rec = Recorder()

        # padded lenngths
        ref_len_c = len(rseq) + 1
        qry_len_c = len(qseq) + 1
        ref_bytes = b'\x00' + rseq.encode('ascii')
        qry_bytes = b'\x00' + qseq.encode('ascii')
        qry_lenD = qry_len_c + ref_len_c - 1

        penalties = self._ffi.new("const struct Penalties*", list(pen))[0]
        best_cell = self._ffi.new("struct bestCell *", [0, 0, 0]) # iniatilize bestCell to 0
        H_buf = self._ffi.new("int16_t[]", qry_lenD * ref_len_c)
        E_buf = self._ffi.new("int16_t[]", qry_lenD * ref_len_c)
        F_buf = self._ffi.new("int16_t[]", qry_lenD * ref_len_c)

        with pair_rec.timed("smith_waterman.dp_fill"):
            self._lib.alignOneNpar(
                ref_len_c, qry_len_c, penalties, ref_bytes, qry_bytes,
                H_buf, E_buf, F_buf, best_cell,
                self.lanes,
                float_counters, N_FLOAT_COUNTERS,
                int_counters, N_INT_COUNTERS,
            )
        # Unless this code is changed nothing is done with float_counters and int_counters

        self.results.append(AlignmentResult(best_cell.score, best_cell.row, best_cell.col))
        self.pair_recs.append(pair_rec)


        # Record final corrected H values for this column as h_matrix cell events.
        # It's just a copy into a Recorder
        for i in range(ref_len_c):
            for j in range(qry_len_c):
                # account for diagonally stored H so traverse diagonally
                pair_rec.add_cell_event("h_matrix", j, i, H_buf[(j+i)*ref_len_c+i])

        self.rec.add_time("smith_waterman.dp_fill", pair_rec.times.get("smith_waterman.dp_fill", 0.0))

    def _needs_backtrace(self, pair_index: int) -> bool:
        """Decide whether pair_index should get a real second-pass backtrace.

        STUB: the real condition (e.g. score above a threshold, or some other
        selection the caller cares about) isn't implemented yet, so every
        pair currently qualifies. Once a condition exists, only it should
        change here -- _second_pass_backtrace() already handles "some pairs
        skipped" via the None entries it appends for pairs that don't qualify.
        """
        return True

    def _second_pass_backtrace(self, pen: array.array) -> None:
        """Compute a real backtrace for pairs selected by _needs_backtrace().

        The GPU pass only ever produces best_cell (score, row, col) -- H/E/F
        live and die on the device (see _align_batch), so there's nothing on
        the host to walk back through. Until a dedicated GPU traceback kernel
        exists, this falls back to re-running the scalar CPU DP (full H +
        ptr matrices) on just the selected pair and tracing back from there.
        That's redundant work for a single pair, but it's exact, and it's
        only paid for pairs _needs_backtrace() actually selects.
        """
        for index, (result, (_qname, qseq, _rname, rseq)) in enumerate(
            zip(self.results, self.pairs)
        ):
            if not self._needs_backtrace(index):
                self.tracebacks.append(None)
                continue
            pair_rec = Recorder()
            _best, h, ptr = smith_waterman_dp(qseq, rseq, pen, pair_rec)
            tb = traceback_alignment(qseq, rseq, h, ptr, result, pair_rec)
            self.tracebacks.append(tb)
            self.rec.add_time(
                "hip_diagonal.second_pass_backtrace",
                pair_rec.times.get("smith_waterman.dp_fill", 0.0)
                + pair_rec.times.get("smith_waterman.traceback", 0.0),
            )

    def run(self, pen: array.array) -> None:
        self._align_batch(pen)

        if self.second_pass:
            self._second_pass_backtrace(pen)
        else:
            self.tracebacks = [None] * len(self.pairs)

        # This code runs as a previous and working replacement to run(), but loops the multiple
        # sequences through python instead of C calling C's align_one(). As such, it calls
        # a gpu kernel per align, which is ridiculous. This was for practice.
        # for index in range(len(self.pairs)):
        #     self._align_one(index, pen)

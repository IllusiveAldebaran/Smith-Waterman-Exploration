"""C-backed scalar Smith-Waterman via cffi (compiled from scalar.c / swag.h).

pen layout: array.array('b', [match, mismatch, del_open, del_ext, ins_open, ins_ext])
match/mismatch are pre-negated; all six values are passed directly through
as the penalties[6] array.
"""

from __future__ import annotations

import array
import os

from cffi import FFI

from ..types import Aligner, AlignmentResult, Recorder

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
            line for line in f if not line.lstrip().startswith("#") and not line.lstrip().startswith("extern")
        )
    ffi.cdef(_header_decls)

    _DEFAULT_SO_PATH = os.path.join(_here, "libhipdiagonal.so")
    so_path = os.environ.get("SWAG_SCALAR_LIB", _DEFAULT_SO_PATH)

    try:
        lib = ffi.dlopen(so_path)
    except OSError as exc:
        raise ImportError(f"hip_diagonal: failed to load {so_path}") from exc
    return lib, ffi


class HIPDiagonalImpl(Aligner):
    """HIP GPU accelerated Smith-Waterman implementation."""

    def __init__(self, lanes: int = 8, verbose: int = 0) -> None:
        self.verbose = verbose
        self.lanes = lanes
        self.rec = Recorder(verbose=verbose)
        self.results: list[AlignmentResult] = []
        self.pair_recs: list[Recorder] = []
        self._lib, self._ffi = _build_lib()

    def _align_batch(self, pen: array.array) -> None:
        """Align a batch of reference and query pairs via the C kernel.
        Packs all of the sequences and matrices together before passing into C kernel
        Best score of each pair is stored

        Records/Copies back into recorders
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
        ref_len_c = len(self.pairs[0][3]) + 1
        qry_len_c = len(self.pairs[0][1]) + 1
        ref_bytes = b''.join(b'\x00' + str(pair[3]).encode('ascii') for pair in self.pairs)
        qry_bytes = b''.join(b'\x00' + str(pair[1]).encode('ascii') for pair in self.pairs)

        # index by 0 to make it clear we're passing the value, not the pointer.
        penalties = self._ffi.new("const struct Penalties*", list(pen))[0]
        best_cell = self._ffi.new("struct bestCell[]", num_pairs)
        H_buf = self._ffi.new("int16_t[]", num_pairs * qry_len_c * ref_len_c)
        E_buf = self._ffi.new("int16_t[]", num_pairs * qry_len_c * ref_len_c)
        F_buf = self._ffi.new("int16_t[]", num_pairs * qry_len_c * ref_len_c)

        with self.rec.timed("smith_waterman.dp_fill"):
            self._lib.alignBatch(
                num_pairs, ref_len_c, qry_len_c, penalties, ref_bytes, qry_bytes,
                H_buf, E_buf, F_buf, best_cell,
                float_counters, N_FLOAT_COUNTERS,
                int_counters, N_INT_COUNTERS,
            )
        # Another way to bring timings/counters but assuming the C code counted it
        #if N_FLOAT_COUNTERS > 0:
        #    self.rec.add_time("smith_waterman.dp_fill", float_counters[0])



        # Record final corrected H values for this column as h_matrix cell events.
        # It's just a copy into a Recorder
        for np in range(num_pairs):
            pair_rec = Recorder()
            h_offset = np * ref_len_c * qry_len_c
            res_offset =  ref_len_c * qry_len_c
            for i in range(ref_len_c):
                for j in range(qry_len_c):
                    pair_rec.add_cell_event("h_matrix", j, i, H_buf[h_offset+j*ref_len_c+i])
            self.pair_recs.append(pair_rec)
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

    def run(self, pen: array.array) -> None:
        #self._align_batch(pen)

        # This code runs as a previous and working replacement to run(), but loops the multiple 
        # sequences through python instead of C calling C's align_one()
        # we know the size by now, we're starting the references here
        for index in range(len(self.pairs)):
            self._align_one(index, pen)

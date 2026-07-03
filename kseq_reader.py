"""cffi binding for kseq_wrapper.so, a thin extern "C" shim around kseqpp's
SeqStreamIn/KSeq (https://github.com/cartoonist/kseqpp).

Build the shared library once with `make` (see Makefile) before importing
this module. It requires a C++17 compiler and zlib (`-lz`); no Python-side
build step is needed beyond that, since cffi's ABI-mode dlopen() just loads
the already-compiled .so.
"""

from __future__ import annotations

import os
from cffi import FFI

ffi = FFI()

ffi.cdef(
    """
    void* kseq_open(const char* filename);
    int kseq_read(void* handle);
    const char* kseq_get_name(void* handle);
    const char* kseq_get_comment(void* handle);
    const char* kseq_get_seq(void* handle);
    void kseq_close(void* handle);
    """
)

_here = os.path.dirname(os.path.abspath(__file__))
_so_path = os.path.join(_here, "kseq_wrapper.so")

try:
    _lib = ffi.dlopen(_so_path)
except OSError as exc:
    raise ImportError(
        f"kseq_wrapper.so not found or failed to load at {_so_path}. "
        "Build it first with `make` (requires g++ with C++17 support and "
        "zlib)."
    ) from exc


class KseqReader:
    """Iterate (name, seq) pairs from a FASTA/FASTQ file, gzip or plain.

    Wraps kseqpp's SeqStreamIn. Use as a context manager or call close()
    explicitly; the underlying C++ stream and record are heap-allocated
    on the C++ side and must be freed via kseq_close().
    """

    def __init__(self, path: str) -> None:
        self._handle = _lib.kseq_open(path.encode("utf-8"))
        if self._handle == ffi.NULL:
            raise FileNotFoundError(f"kseqpp could not open: {path}")

    def __iter__(self) -> "KseqReader":
        return self

    def __next__(self) -> tuple[str, str]:
        if not _lib.kseq_read(self._handle):
            raise StopIteration
        # Copy the strings out immediately: these const char* values point
        # into a std::string owned by the C++-side KSeq record, and that
        # record is overwritten in place on the *next* kseq_read() call.
        name = ffi.string(_lib.kseq_get_name(self._handle)).decode("utf-8")
        seq = ffi.string(_lib.kseq_get_seq(self._handle)).decode("utf-8")
        return name, seq

    def close(self) -> None:
        if self._handle is not None:
            _lib.kseq_close(self._handle)
            self._handle = None

    def __enter__(self) -> "KseqReader":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

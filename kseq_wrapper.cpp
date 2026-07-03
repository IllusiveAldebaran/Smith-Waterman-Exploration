// kseq_wrapper.cpp
//
// Minimal extern "C" shim around kseqpp's SeqStreamIn/KSeq so it can be
// called from Python via cffi. All C++ objects (SeqStreamIn, KSeq) live
// on the C++ heap behind an opaque handle; Python only ever sees a void*.
//
// Lifetime note: kseq_get_name/seq/comment return a pointer into a
// std::string owned by the handle's KSeq record. That pointer is valid
// only until the *next* call to kseq_read() on the same handle (kseq_read
// overwrites the record in place). Copy the string out (ffi.string())
// before advancing.

#include "kseq++/seqio.hpp"
#include <unistd.h>

using namespace klibpp;

struct KseqHandle {
    SeqStreamIn* stream;
    KSeq rec;
};

extern "C" {

// Opens filename (plain or gzipped FASTA/FASTQ). Returns NULL on failure
// (bad path, unreadable file, etc.) so the Python side can raise cleanly
// instead of segfaulting on a null stream.
void* kseq_open(const char* filename) {
    // kseqpp/gzopen don't fail at open time for a missing or unreadable
    // file -- they only discover it lazily on the first read, at which
    // point they just look like an empty file. Check explicitly here so
    // a bad path fails loudly and immediately instead of silently
    // producing zero records.
    if (access(filename, R_OK) != 0) {
        return nullptr;
    }
    KseqHandle* h = new KseqHandle();
    h->stream = new SeqStreamIn(filename);
    if (!(*h->stream)) {
        delete h->stream;
        delete h;
        return nullptr;
    }
    return static_cast<void*>(h);
}

// Advances to the next record. Returns 1 if a record was read, 0 at
// EOF or on a parse error (truncated quality string, etc.).
int kseq_read(void* handle) {
    KseqHandle* h = static_cast<KseqHandle*>(handle);
    *h->stream >> h->rec;
    return (*h->stream) ? 1 : 0;
}

const char* kseq_get_name(void* handle) {
    return static_cast<KseqHandle*>(handle)->rec.name.c_str();
}

const char* kseq_get_comment(void* handle) {
    return static_cast<KseqHandle*>(handle)->rec.comment.c_str();
}

const char* kseq_get_seq(void* handle) {
    return static_cast<KseqHandle*>(handle)->rec.seq.c_str();
}

void kseq_close(void* handle) {
    KseqHandle* h = static_cast<KseqHandle*>(handle);
    if (h) {
        delete h->stream;
        delete h;
    }
}

} // extern "C"

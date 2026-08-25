#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>

typedef struct Penalties {
  int8_t match, mismatch, delOpen, delExt, insOpen, insExt;
} Penalties;

typedef struct bestCell {
  uint16_t row;   // query end position, 1-indexed (H-matrix convention: row=query, col=reference)
  uint16_t col;   // reference end position, 1-indexed
  uint16_t score;
} bestCell;

typedef struct sswTraceback {
  int32_t refBegin;   // 1-indexed start_reference; only valid when wantTraceback was set
  int32_t qryBegin;   // 1-indexed start_query; only valid when wantTraceback was set
  uint32_t* cigar;    // BAM-style cigar ops (low 4 bits: 0=M/1=I/2=D, high 28 bits: length);
                       // owned by this struct -- free via freeSSWTraceback
  int32_t cigarLen;
} sswTraceback;

// Aligns numAligns (query, reference) pairs using the Complete Striped
// Smith-Waterman Library (SSW) as the underlying kernel. Pairs may have
// different lengths -- refSeqs/qrySeqs are the raw ASCII sequences
// concatenated with no padding; refLens/qryLens give each pair's length in
// the same order.
//
// wantTraceback selects which of ssw_align's internal passes run:
//   0 -- only the forward striped SIMD pass (score + end position). This is
//        the fast path (best_cell_only).
//   1 -- also runs a reverse SIMD pass (begin position) and a banded-DP pass
//        that produces a real CIGAR, written into tracebacks[i]. tracebacks
//        must point to a numAligns-length array when wantTraceback is set,
//        and is ignored (may be NULL) otherwise.
void alignBatchSSW(const Penalties penalties,
                    const char* refSeqs, const int32_t* refLens,
                    const char* qrySeqs, const int32_t* qryLens,
                    int32_t numAligns, int wantTraceback,
                    bestCell* best_cells, sswTraceback* tracebacks);

// Frees the cigar buffers allocated into tracebacks by alignBatchSSW.
// Safe to call with tracebacks == NULL.
void freeSSWTraceback(sswTraceback* tracebacks, int32_t numAligns);

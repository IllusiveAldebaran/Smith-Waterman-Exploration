#include "ssw_wrap.h"
#include "Complete-Striped-Smith-Waterman-Library/src/ssw.h"

#include <string.h>

// Encodes one ASCII base into SSW's numeric alphabet. Anything outside
// ACGT (e.g. 'N' from FASTA input) becomes the ambiguous index (4), which
// alignBatchSSW's substitution matrix scores as 0 against every base --
// same convention as the SSW library's own example.c.
static int8_t encodeBase(char c) {
  switch (c) {
    case 'A': return 0;
    case 'C': return 1;
    case 'G': return 2;
    case 'T': return 3;
    default:  return 4;
  }
}

void alignBatchSSW(const Penalties penalties,
                    const char* refSeqs, const int32_t* refLens,
                    const char* qrySeqs, const int32_t* qryLens,
                    int32_t numAligns, int wantTraceback,
                    bestCell* best_cells, sswTraceback* tracebacks) {
  // 5x5 substitution matrix over {A, C, G, T, ambiguous}. Diagonal = actual
  // match reward, off-diagonal = actual mismatch penalty -- undoing the
  // pre-negated convention used everywhere else in this repo (see
  // score_pair() in scalar.py / the cellScore computation in scalar.c).
  int8_t mat[25];
  {
    int k = 0;
    for (int l = 0; l < 4; l++) {
      for (int m = 0; m < 4; m++) mat[k++] = (l == m) ? -penalties.match : -penalties.mismatch;
      mat[k++] = 0; // ambiguous column
    }
    for (int m = 0; m < 5; m++) mat[k++] = 0; // ambiguous row
  }

  int32_t totalRefLen = 0, totalQryLen = 0;
  for (int32_t i = 0; i < numAligns; i++) {
    totalRefLen += refLens[i];
    totalQryLen += qryLens[i];
  }

  // Numeric-encoded copies of the whole batch's sequences. SSW needs int8_t
  // symbol indices, not ASCII -- encode once here rather than per pair.
  int8_t* refNums = (int8_t*)malloc(totalRefLen > 0 ? totalRefLen : 1);
  int8_t* qryNums = (int8_t*)malloc(totalQryLen > 0 ? totalQryLen : 1);
  for (int32_t i = 0; i < totalRefLen; i++) refNums[i] = encodeBase(refSeqs[i]);
  for (int32_t i = 0; i < totalQryLen; i++) qryNums[i] = encodeBase(qrySeqs[i]);

  // Caller (c_farrar/__init__.py) already validated delOpen==insOpen and
  // delExt==insExt -- SSW only has one affine gap cost, applied to both
  // directions, so there's no separate insertion/deletion weight to pass.
  const uint8_t weightGapO = (uint8_t)penalties.delOpen;
  const uint8_t weightGapE = (uint8_t)penalties.delExt;

  // Per-pair offsets into refNums/qryNums, computed up front rather than as
  // a running accumulator inside the alignment loop below -- a running
  // accumulator is a loop-carried dependency (each iteration needs every
  // prior iteration to have already run), which would make the loop below
  // unsafe to parallelize. With offsets precomputed, every iteration is
  // independent.
  int32_t* refOffs = (int32_t*)malloc(sizeof(int32_t) * (numAligns > 0 ? numAligns : 1));
  int32_t* qryOffs = (int32_t*)malloc(sizeof(int32_t) * (numAligns > 0 ? numAligns : 1));
  {
    int32_t refOff = 0, qryOff = 0;
    for (int32_t i = 0; i < numAligns; i++) {
      refOffs[i] = refOff;
      qryOffs[i] = qryOff;
      refOff += refLens[i];
      qryOff += qryLens[i];
    }
  }

  // Each iteration is now fully independent: its own ssw_init/ssw_align
  // call, reading a disjoint (precomputed-offset) slice of refNums/qryNums,
  // writing only to its own index i of best_cells/tracebacks. ssw.c has no
  // shared mutable state (every s_profile*/s_align* is allocated fresh per
  // call), so this is safe to run across threads. schedule(dynamic) because
  // alignment cost scales with qryLen*refLen, which varies a lot pair to
  // pair -- static chunking would leave some threads idle while others are
  // stuck with the few long pairs. Falls back to a plain sequential loop
  // when built without OpenMP.
  #ifdef _OPENMP
  #pragma omp parallel for schedule(dynamic)
  #endif
  for (int32_t i = 0; i < numAligns; i++) {
    const int8_t* refNum = refNums + refOffs[i];
    const int8_t* qryNum = qryNums + qryOffs[i];
    int32_t refLen = refLens[i];
    int32_t qryLen = qryLens[i];

    // score_size=2: let SSW auto-pick byte vs word lanes per pair instead of
    // risking silent score-clamping at 255 in byte mode (see example.c).
    s_profile* profile = ssw_init(qryNum, qryLen, mat, 5, 2);

    // maskLen only affects sub-optimal-alignment reporting (score2/ref_end2),
    // which callers here don't use; kept >= 15 (the library's own minimum)
    // purely to avoid its stderr warning.
    int32_t maskLen = qryLen / 2 > 15 ? qryLen / 2 : 15;

    // flag=1 ("bit 8" in ssw.h's terms): when wantTraceback, always run the
    // reverse pass + banded-DP cigar pass. flag=0: forward SIMD pass only.
    uint8_t flag = wantTraceback ? 1 : 0;
    s_align* result = ssw_align(profile, refNum, refLen, weightGapO, weightGapE,
                                 flag, 0, 0, maskLen);

    best_cells[i].score = (uint16_t)result->score1;
    best_cells[i].row = (uint16_t)(result->read_end1 + 1); // query, 1-indexed
    best_cells[i].col = (uint16_t)(result->ref_end1 + 1);  // reference, 1-indexed

    if (wantTraceback && tracebacks != NULL) {
      tracebacks[i].refBegin = result->ref_begin1 + 1;
      tracebacks[i].qryBegin = result->read_begin1 + 1;
      tracebacks[i].cigar = result->cigar;   // ownership moves to tracebacks[i]
      tracebacks[i].cigarLen = result->cigarLen;
      result->cigar = NULL; // so the plain free() below doesn't matter either way
    }

    // Not align_destroy(result): that also frees result->cigar, which we've
    // just taken ownership of above when wantTraceback is set. When it's
    // not set, result->cigar is NULL anyway (ssw_align only allocates it
    // along the cigar-generation path), so plain free() is correct either way.
    free(result);
    init_destroy(profile);
  }

  free(refOffs);
  free(qryOffs);
  free(refNums);
  free(qryNums);
}

void freeSSWTraceback(sswTraceback* tracebacks, int32_t numAligns) {
  if (tracebacks == NULL) return;
  for (int32_t i = 0; i < numAligns; i++) free(tracebacks[i].cigar);
}

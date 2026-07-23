#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include "swag.h"
#include <time.h>
#include <stdio.h>

// Don't want multiple implementations to clash
#ifndef SW_IMPLEMENTATION
#define SW_IMPLEMENTATION

#define max(A, B) ((A) > (B) ? (A) : (B))

// align one reference and sequence
void alignOne(const uint16_t refLen, const uint16_t qryLen, const Penalties penalties, const char* refSeq, const char* qrySeq, int16_t* H, int16_t* E, int16_t* F, bestCell* best_cell, float* floatCounters, int nfC, int* intCounters, int niC) {
  const int8_t penMatch     = penalties.match;
  const int8_t penMisMatch  = penalties.mismatch;
  const int8_t penDelOpen   = penalties.delOpen;
  const int8_t penDelExt    = penalties.delExt;
  const int8_t penInsOpen   = penalties.insOpen;
  const int8_t penInsExt    = penalties.insExt;
  // self reminder: deletions are a horizontal thing, along the reference
  // insertions are along the vertical, along the query.

  int16_t cellScore;
  for(size_t j = 1; j < qryLen; j++){
    for(size_t i = 1; i < refLen; i++){
      cellScore = (refSeq[i]==qrySeq[j]) ? -penMatch : -penMisMatch;
      //printf("Comparing %c with %c\n", refSeq[i], qrySeq[j]);
      cellScore += H[(j-1)*refLen+i-1];

      // update E,F,H
      E[j*refLen+i] = max(E[(j  )*refLen+i-1]-penDelExt, H[(j  )*refLen+i-1]-penDelOpen);
      F[j*refLen+i] = max(F[(j-1)*refLen+i  ]-penInsExt, H[(j-1)*refLen+i  ]-penInsOpen);
      //printf("Scores are: E:%d F:%d cellScore:%d \n", E[j*refLen+i], F[j*refLen+i], cellScore);
      H[j*refLen+i] = max(max(cellScore, 0), max(E[j*refLen+i], F[j*refLen+i]));

      if (H[j*refLen+i] > (int16_t)best_cell->score) {
        best_cell->row = (uint16_t)j;
        best_cell->col = (uint16_t)i;
        best_cell->score = (uint16_t)H[j*refLen+i];
      }
    }
  }
}

// align in a batch
void alignBatch(const uint16_t count, const uint16_t refLen, const uint16_t qryLen, const Penalties penalties, const char* refSeq, const char* qrySeq, int16_t* H, int16_t* E, int16_t* F, bestCell* best_cell, float* floatCounters, int nfC, int* intCounters, int niC) {

  // If we select to have our C code time we can use this
  //struct timespec start, end;
  //clock_gettime(CLOCK_MONOTONIC, &start);

  int dp_size = refLen * qryLen;
  int offset = 0;
  const char* single_refSeq = refSeq;
  const char* single_qrySeq = qrySeq;
  int16_t* single_H = H;
  int16_t* single_E = E;
  int16_t* single_F = F;

  for(int i = 0; i < count; i++) {
    alignOne(refLen, qryLen, penalties, single_refSeq, single_qrySeq, single_H, single_E, single_F, &best_cell[i],
        floatCounters, nfC, intCounters, niC);

    single_refSeq += refLen;
    single_qrySeq += qryLen;
    single_H += dp_size;
    single_E += dp_size;
    single_F += dp_size;
  }

  // record elapsed if our array is nonzero
  //clock_gettime(CLOCK_MONOTONIC, &end);
  //float elapsed = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) * 1e-9;
  //if(nfC > 0) floatCounters[0] = elapsed;
}

#endif

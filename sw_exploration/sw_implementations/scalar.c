#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include "swag.h"

// Don't want multiple implementations to clash
#ifndef SW_IMPLEMENTATION
#define SW_IMPLEMENTATION

#define max(A, B) ((A) > (B) ? (A) : (B))

// align one reference and sequence
void alignOne(const uint16_t refLen, const uint16_t qryLen, const int8_t penalties[6], const char* refSeq, const char* qrySeq, int16_t* H, int16_t* E, int16_t* F, bestCell* best_cell) {
  const int8_t penMatch     = penalties[0];
  const int8_t penMisMatch  = penalties[1];
  const int8_t penDelOpen   = penalties[2];
  const int8_t penDelExt    = penalties[3];
  const int8_t penInsOpen   = penalties[4];
  const int8_t penInsExt    = penalties[5];
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

// align one reference and sequence
void alignBatch(const uint16_t count, const uint16_t refLen, const uint16_t qryLen, const int8_t penalties[6], const char* refSeq, const char* qrySeq, int16_t* H, int16_t* E, int16_t* F, bestCell* best_cell) {
  const int8_t penMatch     = penalties[0];
  const int8_t penMisMatch  = penalties[1];
  const int8_t penDelOpen   = penalties[2];
  const int8_t penDelExt    = penalties[3];
  const int8_t penInsOpen   = penalties[4];
  const int8_t penInsExt    = penalties[5];
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
    }
  }

}

#endif

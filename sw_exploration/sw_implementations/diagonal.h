#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <time.h>
#include <stdio.h>

typedef struct Penalties {
  int8_t match, mismatch, delOpen, delExt, insOpen, insExt;
} Penalties;

typedef struct bestCell{
  uint16_t row;
  uint16_t col;
  uint16_t score;
} bestCell;

//void alignOne(const uint16_t refLen, const uint16_t qryLen, const Penalties penalties, const char* refSeq, const char* qrySeq, int16_t* H, int16_t* E, int16_t* F, bestCell* best_cell, float* floatCounters, int nfC, int* intCounters, int niC);

// same as before but has npar for deciding some parallelization at runtime
// Weird formatting because the python cffi implementation manually checks for this syntax to ignore these lines
extern "C" 
{
void alignOneNpar(const uint16_t max_refLen, const uint16_t max_qryLen, const Penalties penalties, const char* refSeq, const char* qrySeq, int16_t* H, int16_t* E, int16_t* F, bestCell* best_cell, int npar, float* floatCounters, int nfC, int* intCounters, int niC);

void alignBatchNpar(const uint16_t max_refLen, const uint16_t max_qryLen, const Penalties penalties, const char* refSeq, const char* qrySeq, int16_t* H, int16_t* E, int16_t* F, bestCell* best_cells, uint32_t numAligns, int npar, float* floatCounters, int nfC, int* intCounters, int niC);
}

/*
 * Assumed that the reference lengths and query lengths are all the same.
 * Note that even if they are not they can be padded.
 */
//void alignBatch(const uint16_t count, const uint16_t refLen, const uint16_t qryLen, const Penalties penalties, const char* refSeq, const char* qrySeq, int16_t* H, int16_t* E, int16_t* F, bestCell* best_cell, float* floatCounters, int nfC, int* intCounters, int niC);

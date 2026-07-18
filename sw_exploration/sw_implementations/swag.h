#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>

typedef struct bestCell{
  uint16_t row;
  uint16_t col;
  uint16_t score;
} bestCell;

void alignOne(const uint16_t refLen, const uint16_t qryLen, const int8_t penalties[6], const char* refSeq, const char* qrySeq, int16_t* H, int16_t* E, int16_t* F, bestCell* best_cell);

/*
 * Assumed that the reference lengths and query lengths are all the same.
 * Note that even if they are not they can be padded.
 */
void alignBatch(const uint16_t count, const uint16_t refLen, const uint16_t qryLen, const int8_t penalties[6], const char* refSeq, const char* qrySeq, int16_t* H, int16_t* E, int16_t* F, bestCell* best_cell);

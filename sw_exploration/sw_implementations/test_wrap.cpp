#include <stdio.h>
#include <string.h>

#include "diagonal.hip"
//#include "scalar.c"

#include "seq_utils.h"

#include <hip/hip_runtime.h>
#include <iostream>
#define HIP_CHECK(expression)                                \
{                                                            \
    const hipError_t err = expression;                       \
    if(err != hipSuccess)                                    \
    {                                                        \
        std::cerr << "HIP error: " << hipGetErrorString(err) \
            << " at " << __LINE__ << "\n";                   \
    }                                                        \
}

/* 
 * Quick test of DP
 *
 */

#define MATCH -2
#define MISMATCH 1
#define PENDELO 3
#define PENDELE 1
#define PENINSO 3
#define PENINSE 1

int main(int argc, char** argv) {
  // argc → number of arguments
  // argv → array of arguments
  
  if (argc != 3) {
    printf("Incorrect number of arguments\n");
    printf("Usage: ./dp_test <reference> <query>\n");
    return 1;
  }

  // we are including the prefixed padding we are going to add
  uint16_t refLen = strlen(argv[1])+1;
  uint16_t qryLen = strlen(argv[2])+1;

  uint16_t qryLenDiagonal = qryLen + refLen - 1;

  struct bestCell best_cell = {0, 0, 0};
  // we are not storing strings with \x00 at the start.
  // This can cause problems if you're trying to print the sequence as a regular c_string..
  // Just know to start at index 1. Null terminator after the stirng is also kept
  // It doesn't affect the code... just may be better for batching later
  // e.g. a regular c string like "AGCT\x00" -> "\x00AGCT\x00"
  char* refSeq = (char*)calloc(refLen+1, sizeof(char)); // notice the padding at the start and at the end
  char* qrySeq = (char*)calloc(qryLen+1, sizeof(char));
  memcpy(refSeq+1, argv[1], refLen);
  memcpy(qrySeq+1, argv[2], qryLen);

  int16_t* H = (int16_t*)calloc(refLen*qryLenDiagonal, sizeof(int16_t));
  int16_t* E = (int16_t*)calloc(refLen*qryLenDiagonal, sizeof(int16_t));
  int16_t* F = (int16_t*)calloc(refLen*qryLenDiagonal, sizeof(int16_t));
  int8_t penalties[6] = {MATCH, MISMATCH, PENDELO, PENDELE, PENINSO, PENINSE};

  float fCount[0];
  int intCount[0];
  alignOne(refLen, qryLen, penalties, refSeq, qrySeq, H, E, F, &best_cell, fCount, 0, intCount, 0);

  //showDP((uint8_t*)(refSeq+1), refLen-1, (uint8_t*)(qrySeq+1), qryLen-1, H, true);

  printf("Best Cell: (%d, %d) diagonal aka (%d, %d) score: %d\n", best_cell.col, best_cell.row, best_cell.col, best_cell.row-best_cell.col, best_cell.score);


  free(H);
  free(E);
  free(F);
  free(refSeq);
  free(qrySeq);
  return 0;
}

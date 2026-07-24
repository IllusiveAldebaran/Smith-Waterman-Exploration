#include <stdio.h>
#include <string.h>
#include <cstdlib>

#include "diagonal.c"

#include "seq_utils.h"

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
  
  if (argc != 4) {
    printf("Incorrect number of arguments\n");
    printf("Usage: ./dp_test <reference> <query> <threads>\n");
    printf("Threads should be <min(reference Length, hardware limit)\n");
    return 1;
  }

  // we are including the prefixed padding we are going to add
  uint16_t refLen = strlen(argv[1])+1;
  uint16_t qryLen = strlen(argv[2])+1;
  int npar = atoi(argv[3]);

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
  // these are created just so it fits the code... but they're not used right now as they don't need to exist outside the GPU at the moment
  int16_t* E;
  int16_t* F;

  Penalties penalties = {MATCH, MISMATCH, PENDELO, PENDELE, PENINSO, PENINSE};
  // just temporary empty variables... for prof/counting
  int nfC = 0;
  int niC = 0;
  float fCount[nfC];
  int intCount[niC];

  alignOneNpar(refLen, qryLen,
	   	penalties,
	   	refSeq, qrySeq,
	   	H, E, F,
	   	&best_cell,
	   	npar,
      	   	fCount, nfC, intCount, niC);
  // fills H matrix and best_cell

  printf("Best Cell: (%d, %d) diagonal aka (%d, %d) score: %d\n", best_cell.col, best_cell.row, best_cell.col, best_cell.row-best_cell.col, best_cell.score);
  //print DP to stdout
  //showDP((uint8_t*)(refSeq+1), refLen-1, (uint8_t*)(qrySeq+1), qryLen-1, H, true);

  free(H);
  free(E);
  free(F);
  free(refSeq);
  free(qrySeq);
  return 0;
}

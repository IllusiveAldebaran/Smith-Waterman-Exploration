#pragma once

#include <cstdint>
#include <cstdlib>
#include <cctype>
#include <fstream>
#include <iostream>
#include <string>
#include <algorithm>

// prints the alignment DP matrix and sequences
// If diagonally aligned then prints out differently
// Sequences assumed to have 0th prefixed padding
void showDP(uint8_t* refSeq, uint32_t refLen, uint8_t* qrySeq, uint32_t qryLen, int16_t* DP, bool diagAligned=false) {

  printf("Showing DP scores (%dx%d):\n", qryLen, refLen);
  
  printf("    ");
  for(int i = 0; i<refLen; i++){
    printf("   %c", refSeq[i]? (char)refSeq[i] : '0');
  }
  printf("\n");
  printf("%c  +", (diagAligned)? '0' : ' ');// first element of a query Sequence is '0' always
  for(int i = 0; i <= refLen; i++) printf("————");
  printf("\n");
  
  if(!diagAligned) {
    fprintf(stderr, "Matrix not diagonally Aligned and code needs to be updated for new indexing\n");
    /*
    for(int j = 0; j<DP_ROWS; j++) {
      if(j != 0)
        printf(" %c |", qrySeq[j-1]); // we are doing one more than needed
      else
        printf("   |");

      for(int i = 0; i<DP_COLS; i++) {
        printf(" %3d", DP[j * DP_COLS + i]);
      }
      printf("\n");
    }
    */
  } else {
    for(int j = 0; j < refLen + qryLen - 1; j++) { // -2 because of prefixed padding messup our true number
      if(j < qryLen-1)
        printf("%c  |", (qrySeq[j+1] == 0) ? '0' : qrySeq[j+1]);
      else if (j == (refLen+qryLen - 2))
        printf("%3d|",j); // print out the last row index
      else 
        printf("   |");

      for(int i = 0; i<refLen; i++) {
        printf(" %3d", DP[j * refLen + i]);
      }
      printf("\n");
    }

  }
  

}

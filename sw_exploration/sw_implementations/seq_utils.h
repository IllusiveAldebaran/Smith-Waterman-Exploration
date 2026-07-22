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
void showDP(uint8_t* refSeq, uint32_t refLen, uint8_t* qrySeq, uint32_t qryLen, int16_t* DP, bool diagAligned=false) {
  const size_t DP_COLS = refLen+1;
  const size_t DP_ROWS = qryLen+1;

  printf("Showing DP scores (%dx%d):\n", qryLen, refLen);
  
  printf("        ");
  for(int i = 0; i<refLen; i++){
    printf("   %c", refSeq[i]);
  }
  printf("\n");
  printf("%c  +", (diagAligned)?(char)qrySeq[0] : ' ');
  for(int i = 0; i <= refLen; i++) printf("————");
  printf("\n");
  
  if(!diagAligned) {
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
  } else {
    for(int j = 0; j<DP_ROWS + DP_COLS - 1; j++) {
      if(j == 0)
        printf("%c  |", qrySeq[j+1]);
      else if(j > 0 && j < DP_ROWS-2)
        printf("%c  |", qrySeq[j+1]);
      else
        printf("   |");

      for(int i = 0; i<DP_COLS; i++) {
        printf(" %3d", DP[j * DP_COLS + i]);
      }
      printf("\n");
    }

  }
  

}

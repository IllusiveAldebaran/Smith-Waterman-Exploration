#include <stdio.h>
#include <string.h>
#include <cstdlib>

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


  // GPU Query
  /*
  int deviceCount;
  if (hipGetDeviceCount(&deviceCount) == hipSuccess) {
      for (int i = 0; i < deviceCount; ++i) {
          hipDeviceProp_t prop;
          if (hipGetDeviceProperties(&prop, i) == hipSuccess)
              std::cout << "Device" << i << prop.name << std::endl;
      }
  }
  */


  int16_t* H = (int16_t*)calloc(refLen*qryLenDiagonal, sizeof(int16_t));
  // GPU pointers
  int16_t* d_H;
  int16_t* d_E;
  int16_t* d_F;
  char* d_refSeq;
  char* d_qrySeq;
  struct bestCell* d_best_cell;
  HIP_CHECK(hipMalloc(&d_H, sizeof(*d_H) * (refLen*qryLenDiagonal)));
  HIP_CHECK(hipMalloc(&d_E, sizeof(*d_E) * (refLen*qryLenDiagonal)));
  HIP_CHECK(hipMalloc(&d_F, sizeof(*d_F) * (refLen*qryLenDiagonal)));
  HIP_CHECK(hipMalloc(&d_refSeq, sizeof(*d_refSeq) * refLen));
  HIP_CHECK(hipMalloc(&d_qrySeq, sizeof(*d_qrySeq) * qryLen));
  HIP_CHECK(hipMalloc(&d_best_cell, sizeof(*d_best_cell)));

  // Copy Sequences, only prefixed padding
  HIP_CHECK(hipMemcpy(d_refSeq, refSeq, (refLen) * sizeof(char), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_qrySeq, qrySeq, (qryLen) * sizeof(char), hipMemcpyHostToDevice));
  // this might be unecessary if we can initialize d_best_cell to all 0
  HIP_CHECK(hipMemcpy(d_best_cell, &best_cell, sizeof(bestCell), hipMemcpyHostToDevice));

  Penalties penalties = {MATCH, MISMATCH, PENDELO, PENDELE, PENINSO, PENINSE};

  float fCount[0];
  int intCount[0];

  // KERNEL CALL
  size_t sharedMemBytes = npar * sizeof(struct bestCell); 
  alignOne<<<1, npar, sharedMemBytes>>>(refLen, qryLen, penalties, d_refSeq, d_qrySeq, d_H, d_E, d_F, d_best_cell);//, fCount, 0, intCount, 0);
  HIP_CHECK(hipDeviceSynchronize());

  HIP_CHECK(hipMemcpy(H, d_H, refLen*qryLenDiagonal * sizeof(int16_t), hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(&best_cell, d_best_cell, sizeof(bestCell), hipMemcpyDeviceToHost));

  printf("Best Cell: (%d, %d) diagonal aka (%d, %d) score: %d\n", best_cell.col, best_cell.row, best_cell.col, best_cell.row-best_cell.col, best_cell.score);
  //print DP to stdout
  //showDP((uint8_t*)(refSeq+1), refLen-1, (uint8_t*)(qrySeq+1), qryLen-1, H, true);

  HIP_CHECK(hipFree(d_H));
  HIP_CHECK(hipFree(d_E));
  HIP_CHECK(hipFree(d_F));

  HIP_CHECK(hipFree(d_refSeq));
  HIP_CHECK(hipFree(d_qrySeq));

  free(H);
  free(refSeq);
  free(qrySeq);
  return 0;
}

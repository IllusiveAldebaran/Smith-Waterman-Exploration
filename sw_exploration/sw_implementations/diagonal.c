#include "diagonal.hip"

void alignOneNpar(const uint16_t refLen, const uint16_t qryLen, const Penalties penalties, const char* refSeq, const char* qrySeq, int16_t* H, int16_t* E, int16_t* F, bestCell* best_cell, int npar, float* floatCounters, int nfC, int* intCounters, int niC){
  
  uint16_t qryLenDiagonal = qryLen + refLen - 1;

  // GPU Query
  /*
  int deviceCount;
  if (hipGetDeviceCount(&deviceCount) == hipSuccess) {
      for (int i = 0; i < deviceCount; ++i) {
          hipDeviceProp_t prop;
          if (hipGetDeviceProperties(&prop, i) == hipSuccess)
              printf("Device %d %s\n", i, prop.name);
      }
  }
  */


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
  /*
  printf("Comparing: ");
  for(size_t i = 0; i<refLen; i++) printf("%c", refSeq[i]);
  printf("\nTo \n");
  for(size_t i = 0; i<qryLen; i++) printf("%c", qrySeq[i]);
  printf("\n");
  */

  // KERNEL CALL
  size_t sharedMemBytes = npar * sizeof(struct bestCell); 
  alignOne<<<1, npar, sharedMemBytes>>>(refLen, qryLen, penalties, d_refSeq, d_qrySeq, d_H, d_E, d_F, d_best_cell);//, fCount, 0, intCount, 0);
  HIP_CHECK(hipDeviceSynchronize());

  HIP_CHECK(hipMemcpy(H, d_H, refLen*qryLenDiagonal * sizeof(int16_t), hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(best_cell, d_best_cell, sizeof(bestCell), hipMemcpyDeviceToHost));
  //printf("Best Cell: (%d, %d) diagonal aka (%d, %d) score: %d\n", best_cell->col, best_cell->row, best_cell->col, best_cell->row-best_cell->col, best_cell->score);

  HIP_CHECK(hipFree(d_H));
  HIP_CHECK(hipFree(d_E));
  HIP_CHECK(hipFree(d_F));

  HIP_CHECK(hipFree(d_refSeq));
  HIP_CHECK(hipFree(d_qrySeq));
}

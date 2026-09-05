#include "diagonal.hip"

void alignBatchNpar_ScoreOnly(const uint16_t max_refLen, const uint16_t max_qryLen, const Penalties penalties, const char* refSeq, const char* qrySeq, int16_t* best_scores, uint32_t numAligns, int npar)
{
  
  if(max_refLen > 1024) 
  {
    fprintf(stderr, "One thread per max reference length. Hardware limit is 1024. (max_refLen=%d)", max_refLen);
  }
  // total number of rows in our weird setup
  size_t qryLenDiagonal = (size_t)numAligns * (max_qryLen + 1) + (max_refLen + 1) - 1;

  // GPU Query
  int deviceCount;
  if (hipGetDeviceCount(&deviceCount) == hipSuccess) {
      for (int i = 0; i < deviceCount; ++i) {
          hipDeviceProp_t prop;
          if (hipGetDeviceProperties(&prop, i) == hipSuccess)
              printf("Device %d %s\n", i, prop.name);
      }
  }

  hipDeviceProp_t props;
  HIP_CHECK(hipGetDeviceProperties(&props, 0));
  int compute_units = props.multiProcessorCount;
  int warpSize = props.warpSize;

  // GPU pointers
  char* d_refSeq;
  char* d_qrySeq;
  int16_t* d_best_scores;
  HIP_CHECK(hipMalloc(&d_refSeq, sizeof(*d_refSeq) * numAligns * (max_refLen+1)));
  HIP_CHECK(hipMalloc(&d_qrySeq, sizeof(*d_qrySeq) * numAligns * (max_qryLen+1)));
  HIP_CHECK(hipMalloc(&d_best_scores, numAligns * sizeof(*d_best_scores)));

  // Copy Sequences, only prefixed padding
  HIP_CHECK(hipMemcpy(d_refSeq, refSeq, numAligns*(max_refLen+1) * sizeof(char), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_qrySeq, qrySeq, numAligns*(max_qryLen+1) * sizeof(char), hipMemcpyHostToDevice));
  // this might be unecessary if we can initialize to just 0s
  HIP_CHECK(hipMemcpy(d_best_scores, best_scores, numAligns*sizeof(int16_t), hipMemcpyHostToDevice));

  /*
  printf("Comparing: ");
  for(size_t i = 0; i<numAligns*max_refLen; i++) printf("%c", (refSeq[i] == 0) ? '0' : refSeq[i]);
  printf("\nTo: ");
  for(size_t i = 0; i<numAligns*max_qryLen; i++) printf("%c", (qrySeq[i] == 0) ? '0' : qrySeq[i]);
  printf("\n");
  */

  // KERNEL CALL
  // Warp collective function used for interwarp reduction
  // but shared memory is needed for intrawarp reduction within the same block
  // Finally shared memory is a place to store the best scores before writing it all to global mem
  size_t GRID_SIZE = BLOCK_CU_RATIO*compute_units;
  size_t wavesInWavefront = (npar+warpSize-1)/warpSize;
  size_t fixedSharedBytes = wavesInWavefront * sizeof(int16_t);

  // Check if kernel's shared memory fits on CU
  if (fixedSharedBytes >= props.sharedMemPerBlock) {
    // pathological: npar alone doesn't fit in one block's shared memory.
    // Not fixable by adding blocks -- would need a smaller --lanes value.
    fprintf(stderr, "alignBatchNpar: npar=%d alone needs %zu bytes of shared memory, "
           "device only has %zu bytes per block -- reduce --lanes\n"
           "Reasons could be big difference reference and query lengths\n",
           npar, fixedSharedBytes, (size_t)props.sharedMemPerBlock);
  } else {
    size_t maxAlignsPerBlock = (props.sharedMemPerBlock - fixedSharedBytes) / sizeof(int16_t);
    size_t minGridForSharedMem = ((size_t)numAligns + maxAlignsPerBlock - 1) / maxAlignsPerBlock;
    if (minGridForSharedMem > GRID_SIZE) {
      GRID_SIZE = minGridForSharedMem;
    }
  }

  // Must match alignBatch's own ALIGNS_PER_BLOCK formula in diagonal.hip exactly.
  size_t aligns_per_block = ((size_t)numAligns + GRID_SIZE - 1) / GRID_SIZE;
  size_t alignsBytes = aligns_per_block * sizeof(int16_t);
  size_t mesh_sharing_bytes = 2 * wavesInWavefront * sizeof(int16_t); // shiftScores() needs shared memory to share in between warps. 
  size_t sharedMemBytes = fixedSharedBytes + alignsBytes + mesh_sharing_bytes;
  alignBatch_ScoreOnly<<<GRID_SIZE, npar, sharedMemBytes>>>(max_refLen, max_qryLen, penalties, d_refSeq, d_qrySeq, d_best_scores, numAligns, npar);
  HIP_CHECK(hipDeviceSynchronize());

  HIP_CHECK(hipMemcpy(best_scores, d_best_scores, numAligns*sizeof(int16_t), hipMemcpyDeviceToHost));

  HIP_CHECK(hipFree(d_refSeq));
  HIP_CHECK(hipFree(d_qrySeq));
  HIP_CHECK(hipFree(d_best_scores));
}

void alignBatchNpar_CellOnly(const uint16_t max_refLen, const uint16_t max_qryLen, const Penalties penalties, const char* refSeq, const char* qrySeq, bestCell* best_cells, uint32_t numAligns, int npar){
  
  if(max_refLen > 1024) 
  {
    fprintf(stderr, "One thread per max reference length. Hardware limit is 1024. (max_refLen=%d)", max_refLen);
  }
  // total number of rows in our weird setup
  size_t qryLenDiagonal = (size_t)numAligns * (max_qryLen + 1) + (max_refLen + 1) - 1;

  // GPU Query
  int deviceCount;
  if (hipGetDeviceCount(&deviceCount) == hipSuccess) {
      for (int i = 0; i < deviceCount; ++i) {
          hipDeviceProp_t prop;
          if (hipGetDeviceProperties(&prop, i) == hipSuccess)
              printf("Device %d %s\n", i, prop.name);
      }
  }

  hipDeviceProp_t props;
  HIP_CHECK(hipGetDeviceProperties(&props, 0));
  int compute_units = props.multiProcessorCount;
  int warpSize = props.warpSize;

  // GPU pointers
  char* d_refSeq;
  char* d_qrySeq;
  struct bestCell* d_best_cells;
  HIP_CHECK(hipMalloc(&d_refSeq, sizeof(*d_refSeq) * numAligns * (max_refLen+1)));
  HIP_CHECK(hipMalloc(&d_qrySeq, sizeof(*d_qrySeq) * numAligns * (max_qryLen+1)));
  HIP_CHECK(hipMalloc(&d_best_cells, numAligns * sizeof(*d_best_cells)));

  // Copy Sequences, only prefixed padding
  HIP_CHECK(hipMemcpy(d_refSeq, refSeq, numAligns*(max_refLen+1) * sizeof(char), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_qrySeq, qrySeq, numAligns*(max_qryLen+1) * sizeof(char), hipMemcpyHostToDevice));
  // this might be unecessary if we can initialize d_best_cells to all 0
  HIP_CHECK(hipMemcpy(d_best_cells, best_cells, numAligns*sizeof(struct bestCell), hipMemcpyHostToDevice));

  /*
  printf("Comparing: ");
  for(size_t i = 0; i<numAligns*max_refLen; i++) printf("%c", (refSeq[i] == 0) ? '0' : refSeq[i]);
  printf("\nTo: ");
  for(size_t i = 0; i<numAligns*max_qryLen; i++) printf("%c", (qrySeq[i] == 0) ? '0' : qrySeq[i]);
  printf("\n");
  */

  // KERNEL CALL
  // Shared memory holds shared_best (npar entries of bestCellTall, reduction
  // scratch) and final_best (this block's own share of alignments, as
  // bestCell -- see diagonal.hip, final_best is indexed relative to each
  // block's own alignIndex range, not the whole batch's).
  //    TODO: Changed to interwarp reductions and reduce shared memory for each thread
  size_t GRID_SIZE = BLOCK_CU_RATIO*compute_units;
  size_t fixedSharedBytes = npar * sizeof(struct bestCellTall);

  // Check if kernel's shared memory fits on CU
  if (fixedSharedBytes >= props.sharedMemPerBlock) {
    // pathological: npar alone doesn't fit in one block's shared memory.
    // Not fixable by adding blocks -- would need a smaller --lanes value.
    fprintf(stderr, "alignBatchNpar: npar=%d alone needs %zu bytes of shared memory, "
           "device only has %zu bytes per block -- reduce --lanes\n"
           "Probably due to big difference in reference and query lengths\n",
           npar, fixedSharedBytes, (size_t)props.sharedMemPerBlock);
  } else {
    size_t maxAlignsPerBlock = (props.sharedMemPerBlock - fixedSharedBytes) / sizeof(struct bestCell);
    size_t minGridForSharedMem = ((size_t)numAligns + maxAlignsPerBlock - 1) / maxAlignsPerBlock;
    if (minGridForSharedMem > GRID_SIZE) {
      GRID_SIZE = minGridForSharedMem;
    }
  }

  // Must match alignBatch's own ALIGNS_PER_BLOCK formula in diagonal.hip exactly.
  size_t aligns_per_block = ((size_t)numAligns + GRID_SIZE - 1) / GRID_SIZE;
  size_t alignsBytes = aligns_per_block * sizeof(struct bestCell);
  size_t mesh_sharing_bytes = 2 * sizeof(int16_t) * ((max_refLen+warpSize-1) / warpSize); // shiftScores() needs shared memory to share in between warps. 
  size_t sharedMemBytes = fixedSharedBytes + alignsBytes + mesh_sharing_bytes;
  alignBatch_CellOnly<<<GRID_SIZE, npar, sharedMemBytes>>>(max_refLen, max_qryLen, penalties, d_refSeq, d_qrySeq, d_best_cells, numAligns);//, fCount, 0, intCount, 0);
  HIP_CHECK(hipDeviceSynchronize());

  HIP_CHECK(hipMemcpy(best_cells, d_best_cells, numAligns*sizeof(bestCell), hipMemcpyDeviceToHost));
  //printf("Best Cell: (%d, %d) diagonal aka (%d, %d) score: %d\n", best_cell->col, best_cell->row, best_cell->col, best_cell->row-best_cell->col, best_cell->score);

  HIP_CHECK(hipFree(d_refSeq));
  HIP_CHECK(hipFree(d_qrySeq));
  HIP_CHECK(hipFree(d_best_cells));
}

void alignBatchNpar(const uint16_t max_refLen, const uint16_t max_qryLen, const Penalties penalties, const char* refSeq, const char* qrySeq, bestCell* best_cells, uint32_t numAligns, int npar, float* floatCounters, int nfC, int* intCounters, int niC){
  
  // total number of rows in our weird setup
  size_t qryLenDiagonal = (size_t)numAligns * (max_qryLen + 1) + (max_refLen + 1) - 1;

  // GPU Query
  int deviceCount;
  if (hipGetDeviceCount(&deviceCount) == hipSuccess) {
      for (int i = 0; i < deviceCount; ++i) {
          hipDeviceProp_t prop;
          if (hipGetDeviceProperties(&prop, i) == hipSuccess)
              printf("Device %d %s\n", i, prop.name);
      }
  }

  hipDeviceProp_t props;
  HIP_CHECK(hipGetDeviceProperties(&props, 0));
  int compute_units = props.multiProcessorCount;

  // GPU pointers
  int16_t* d_H;
  int16_t* d_E;
  int16_t* d_F;
  char* d_refSeq;
  char* d_qrySeq;
  struct bestCell* d_best_cells;
  HIP_CHECK(hipMalloc(&d_H, sizeof(*d_H) * ((max_refLen+1)*qryLenDiagonal)));
  HIP_CHECK(hipMalloc(&d_E, sizeof(*d_E) * ((max_refLen+1)*qryLenDiagonal)));
  HIP_CHECK(hipMalloc(&d_F, sizeof(*d_F) * ((max_refLen+1)*qryLenDiagonal)));
  HIP_CHECK(hipMalloc(&d_refSeq, sizeof(*d_refSeq) * numAligns * (max_refLen+1)));
  HIP_CHECK(hipMalloc(&d_qrySeq, sizeof(*d_qrySeq) * numAligns * (max_qryLen+1)));
  HIP_CHECK(hipMalloc(&d_best_cells, numAligns * sizeof(*d_best_cells)));

  // Copy Sequences, only prefixed padding
  HIP_CHECK(hipMemcpy(d_refSeq, refSeq, numAligns*(max_refLen+1) * sizeof(char), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_qrySeq, qrySeq, numAligns*(max_qryLen+1) * sizeof(char), hipMemcpyHostToDevice));
  // this might be unecessary if we can initialize d_best_cells to all 0
  HIP_CHECK(hipMemcpy(d_best_cells, best_cells, numAligns*sizeof(struct bestCell), hipMemcpyHostToDevice));

  /*
  printf("Comparing: ");
  for(size_t i = 0; i<numAligns*max_refLen; i++) printf("%c", (refSeq[i] == 0) ? '0' : refSeq[i]);
  printf("\nTo: ");
  for(size_t i = 0; i<numAligns*max_qryLen; i++) printf("%c", (qrySeq[i] == 0) ? '0' : qrySeq[i]);
  printf("\n");
  */

  // KERNEL CALL
  // Shared memory holds shared_best (npar entries of bestCellTall, reduction
  // scratch) and final_best (this block's own share of alignments, as
  // bestCell -- see diagonal.hip, final_best is indexed relative to each
  // block's own alignIndex range, not the whole batch's).
  //    TODO: Changed to interwarp reductions and reduce shared memory for each thread
  size_t GRID_SIZE = BLOCK_CU_RATIO*compute_units;
  size_t fixedSharedBytes = npar * sizeof(struct bestCellTall);
  if (fixedSharedBytes >= props.sharedMemPerBlock) {
    // pathological: npar alone doesn't fit in one block's shared memory.
    // Not fixable by adding blocks -- would need a smaller --lanes value.
    fprintf(stderr, "alignBatchNpar: npar=%d alone needs %zu bytes of shared memory, "
           "device only has %zu bytes per block -- reduce --lanes\n"
           "Probably due to big difference in reference and query lengths\n",
           npar, fixedSharedBytes, (size_t)props.sharedMemPerBlock);
  } else {
    // final_best used to be sized for the WHOLE batch (numAligns) in every
    // block's shared memory, which silently overflowed the device's
    // per-block shared memory limit for large numAligns (and wasn't even
    // checked -- an oversized launch would return zeroed scores, not an
    // error). Since each block only ever writes its own alignIndex range,
    // final_best only needs to hold that range. If the "ideal" occupancy
    // grid (BLOCK_CU_RATIO*compute_units) would still make each block's
    // share too big to fit, grow the grid until it does -- more, smaller
    // blocks, not a bigger buffer.
    size_t maxAlignsPerBlock = (props.sharedMemPerBlock - fixedSharedBytes) / sizeof(struct bestCell);
    size_t minGridForSharedMem = ((size_t)numAligns + maxAlignsPerBlock - 1) / maxAlignsPerBlock;
    if (minGridForSharedMem > GRID_SIZE) {
      GRID_SIZE = minGridForSharedMem;
    }
  }
  // Must match alignBatch's own ALIGNS_PER_BLOCK formula in diagonal.hip exactly.
  size_t ALIGNS_PER_BLOCK = ((size_t)numAligns + GRID_SIZE - 1) / GRID_SIZE;
  size_t sharedMemBytes = fixedSharedBytes + ALIGNS_PER_BLOCK * sizeof(struct bestCell);
  alignBatch<<<GRID_SIZE, npar, sharedMemBytes>>>(max_refLen, max_qryLen, penalties, d_refSeq, d_qrySeq, d_H, d_E, d_F, d_best_cells, numAligns);//, fCount, 0, intCount, 0);
  HIP_CHECK(hipDeviceSynchronize());

  HIP_CHECK(hipMemcpy(best_cells, d_best_cells, numAligns*sizeof(bestCell), hipMemcpyDeviceToHost));
  //printf("Best Cell: (%d, %d) diagonal aka (%d, %d) score: %d\n", best_cell->col, best_cell->row, best_cell->col, best_cell->row-best_cell->col, best_cell->score);

  HIP_CHECK(hipFree(d_H));
  HIP_CHECK(hipFree(d_E));
  HIP_CHECK(hipFree(d_F));

  HIP_CHECK(hipFree(d_refSeq));
  HIP_CHECK(hipFree(d_qrySeq));
  HIP_CHECK(hipFree(d_best_cells));
}

void alignOneNpar(const uint16_t refLen, const uint16_t qryLen, const Penalties penalties, const char* refSeq, const char* qrySeq, int16_t* H, int16_t* E, int16_t* F, bestCell* best_cell, int npar, float* floatCounters, int nfC, int* intCounters, int niC){
  
  size_t qryLenDiagonal = (size_t)qryLen + refLen - 1;

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

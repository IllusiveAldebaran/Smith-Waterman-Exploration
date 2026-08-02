#include "diagonal.h"

#include <hip/hip_runtime.h>

#define HIP_CHECK(expression)                                \
{                                                            \
    const hipError_t err = expression;                       \
    if(err != hipSuccess)                                    \
    {                                                        \
        fprintf(stderr, "HIP error: %s at %d \n", hipGetErrorString(err) \
            , __LINE__);				     \
    }                                                        \
}

/* Diagonally prealigned
 * Changes the algorithm in memory accesses, but it is essentially the same.
 */
__global__ void alignBatch(const uint16_t max_refLen, const uint16_t max_qryLen,
			 const Penalties penalties,
			 const char* refSeq, const char* qrySeq,
			 int16_t* H, int16_t* E, int16_t* F,
			 bestCell* best_cells, uint32_t numAligns) {
			 // float* floatCounters, int nfC,
			 // int* intCounters, int niC) {

  // g_idx = threadIdx.x + blockIdx.x * blockDim.x;
  int tid = threadIdx.x;
  // we're going to treat refLens and qryLens as if their prefixed padding is part of the deal
  uint16_t refLen = max_refLen+1;
  uint16_t qryLen = max_qryLen+1;

  int alignIndex = 0; // We must keep our offset of our refIndex at all times
  //int refIndex = 0; // Not necessary to define one as it moves with alignIndex
  size_t dp_rows = refLen + (numAligns * qryLen) - 1;

  // Threads should be aligned as length of reference
  if (tid > refLen) {
      printf("Thread ID %d beyond reflen: %d", tid, refLen);
      return;
  };

  // Space to store best_cell for reduction after every thread calculates a thread-local best cell
  extern __shared__ char smem[];

  // For this reason shared_best is our reduction space for work.
  bestCell* shared_best = reinterpret_cast<bestCell*>(smem);
  // Buffer used to copy the best of every align until the end of the kernel. Keeps track of the best cells for each alignment
  bestCell* final_best = reinterpret_cast<bestCell*>(smem + (blockDim.x * sizeof(struct bestCell)));


  // Each thread needs a way to keep track of which alignment it is on since there is overlap
  bestCell thread_best_cell[2] = {{0, 0, 0},{0,0,0}};
 
  const int8_t penMatch     = penalties.match;
  const int8_t penMisMatch  = penalties.mismatch;
  const int8_t penDelOpen   = penalties.delOpen;
  const int8_t penDelExt    = penalties.delExt;
  const int8_t penInsOpen   = penalties.insOpen;
  const int8_t penInsExt    = penalties.insExt;

  
  for(size_t dy = 2; dy < dp_rows; dy++) {

    // Bounds for the initial and final paddings, these should be constants tbh
    size_t dx_lo = (dy >= numAligns * qryLen) ? (dy - (numAligns * qryLen) + 1) : 1;
    size_t dx_hi = (dy < refLen) ? (dy - 1) : (refLen - 1);
    // since we are interleaving references we need to keep track of which one we're on

    // Control variable
    size_t qryInd = dy-dx_lo; // The index of our dy character
    size_t refOff = refLen*alignIndex;

    // iterate through all cols but select to do run the alignment algorithm or not.
    // Makes vectorization (and possibly later on batched alignment) possible
    // check beyond diagonal.
    // Example in (dy, dx) (2, 3) does not exist as its padded area
    //printf("Traversing from col: %zu to %zu row:%zu\n", dx_lo, dx_hi, dy); 
    for(size_t dx = dx_lo + tid; dx <= dx_hi; dx += blockDim.x) {
      qryInd = dy-dx;
      refOff = (qryInd/qryLen)*refLen; // need to check when we're looking at the next sequence
      size_t curAlignIndex = qryInd/qryLen; // like alignIndex but the one this cell is currently on, not which pair we're actively completing
      size_t refInd = dx + refOff; // sometimes we need to move by refLen forward

      int16_t cellScore = (refSeq[refInd]==qrySeq[qryInd]) ? -penMatch : -penMisMatch;
      //printf("Comparing %c with %c, %hd match!\n", refSeq[dx], qrySeq[qryInd], cellScore);

      // small guardrail to check if it's the first 'row' in this alignment, since the 'diagonal' value would be a previous batch
      cellScore += (qrySeq[qryInd-1] != 0) ? H[(dy-2)*refLen+dx-1] : 0;


      // update E,F,H
      E[dy*refLen+dx] = std::max(E[(dy-1)*refLen+dx-1]-penDelExt, H[(dy-1)*refLen+dx-1]-penDelOpen);
      F[dy*refLen+dx] = std::max(F[(dy-1)*refLen+dx  ]-penInsExt, H[(dy-1)*refLen+dx  ]-penInsOpen);
      //printf("Scores are: E:%d F:%d cellScore:%d \n", E[dy*refLen+dx], F[dy*refLen+dx], cellScore);
      //printf("Scores at (%zu, %zu) : (H,E,F) (%d, %d, %d)\n", dx, dy, cellScore, F[dy*refLen+dx], F[dy*refLen+dx]);
      H[dy*refLen+dx] = std::max(std::max(cellScore, (int16_t)0), std::max(E[dy*refLen+dx], F[dy*refLen+dx]));

      
      // stupid guard rail two see if we are in padding. Overwrites a garbage scores to reset them all to 0 for next batch
      if(refSeq[refInd] == 0 || qrySeq[qryInd] == 0) {
        //printf("We're in padding all scores 0's out at (%zu,%zu) refSeq[%zu] qrySeq[%zu]\n", dx, dy, refInd, dy);
        H[dy*refLen+dx] = 0;
        E[dy*refLen+dx] = 0;
        F[dy*refLen+dx] = 0;
      }

      // Check to see if it's the best alignment. If its better, then record it.
      {
        // alignment Double Buffer Index
        int alignDBI = curAlignIndex % 2;

        if (H[dy*refLen+dx] > (int16_t)thread_best_cell[alignDBI].score) {
          thread_best_cell[alignDBI].row = (uint16_t)dy;
          thread_best_cell[alignDBI].col = (uint16_t)dx;
          thread_best_cell[alignDBI].score = (uint16_t)H[dy*refLen+dx];
          //printf("New thread max score: %d at (%d, %d) aka (%d, %d) alignDBI:%d\n", thread_best_cell[alignDBI].score, thread_best_cell[alignDBI].col, thread_best_cell[alignDBI].row, thread_best_cell[alignDBI].col, thread_best_cell[alignDBI].row-thread_best_cell[alignDBI].col, alignDBI);
        }
      }
    }

    // check if we're at a reduction point because an alignment is done
    // So we should perform reduction
    if(dy > (refLen-1) &&  // The first qryLen rows are kind of like an offset. Sorry this isn't a place to be explaining it.
	( (dy+1) - (refLen-1) ) % qryLen == 0) { // give a qryLen offSet to dy because we finish an align every refLen after qryLen rows
      // Since we're already in the next pair we do + 1 for the opposite buffer
      int alignDBI = (( (dy+1)-(refLen-1) )/qryLen % 2) ^ 1;
      //printf("alignDBI %d at (%zu, %zu)\n", alignDBI, dy, dx_lo);
      //printf("Performing Reduction at index %zu shared buf index %d\n", dy, alignDBI);
      //printf("qryInd: %zu qryLen:%d  refLen:%d \n", qryInd, qryLen, refLen);

      // Read input from front buffer to shared
      shared_best[tid] = thread_best_cell[alignDBI];
      __syncthreads();
      //printf("Put this score in shared at thread[%d]: %d at (%d, %d) aka (%d, %d) alignDBI:%d\n", tid, thread_best_cell[alignDBI].score, thread_best_cell[alignDBI].col, thread_best_cell[alignDBI].row, thread_best_cell[alignDBI].col, (thread_best_cell[alignDBI].row - alignIndex * qryLen)-thread_best_cell[alignDBI].col, alignDBI);
      
      // Naive Shared reduction
      for (uint32_t i = 1; i < blockDim.x; i *= 2) {
        if (tid % (2 * i) == 0
           && tid + i < blockDim.x) { // this is an optional check so we can launch with thread count not a power of 2
          if(shared_best[tid].score < shared_best[tid+i].score) {
            shared_best[tid] = shared_best[tid+i];
          }
        }
        __syncthreads();
      }
      __syncthreads();
      // save the best thread into shared space for best alignment per pair
      if(tid == 0) {
        final_best[alignIndex] = shared_best[0];
      }

      __syncthreads();
      // need to wipe this slot back to 0
      shared_best[tid].score = 0;
      thread_best_cell[alignDBI].score = 0;

      // all threads are complete with this alignment pair
      alignIndex++;

      __syncthreads();
    }
  }


  // Write results from shared to back main buffer
  if (tid == 0) {
    for(size_t i = 0; i < numAligns; i++) {
      best_cells[i] = final_best[i];
      //printf("HIP Final %zu: %d at (%d, %d) aka (%d, %d)\n", i, best_cells[i].score, best_cells[i].col, best_cells[i].row, best_cells[i].col, (best_cells[i].row-i*qryLen)-best_cells[i].col);
    }
  }
}


/* Diagonally prealigned
 * Changes the algorithm in memory accesses, but it is essentially the same.
 */

__global__ void alignOne(const uint16_t refLen, const uint16_t qryLen,
			 const Penalties penalties,
			 const char* refSeq, const char* qrySeq,
			 int16_t* H, int16_t* E, int16_t* F,
			 bestCell* best_cell) {
			 // float* floatCounters, int nfC,
			 // int* intCounters, int niC) {

  // g_idx = threadIdx.x + blockIdx.x * blockDim.x;
  int tid = threadIdx.x;

  // Threads should be aligned as length of reference
  if (tid > refLen) {
      printf("Thread ID %d beyond reflen: %d", tid, refLen);
      return;
  };

  // Space to store best_cell for reduction after every thread calculates a thread-local best cell
  extern __shared__ bestCell shared_best[];

  // penalties
  bestCell thread_best_cell = {0, 0, 0};
 
  const int8_t penMatch     = penalties.match;
  const int8_t penMisMatch  = penalties.mismatch;
  const int8_t penDelOpen   = penalties.delOpen;
  const int8_t penDelExt    = penalties.delExt;
  const int8_t penInsOpen   = penalties.insOpen;
  const int8_t penInsExt    = penalties.insExt;

  
  for(size_t dy = 2; dy < refLen + qryLen - 1; dy++) {
    size_t dx_lo = (dy >= qryLen) ? (dy - qryLen + 1) : 1;
    size_t dx_hi = (dy < refLen) ? (dy - 1) : (refLen - 1);

    // iterate through all cols but select to do run the alignment algorithm or not.
    // Makes vectorization (and possibly later on batched alignment) possible
    // check beyond diagonal.
    // Example in (dy, dx) (2, 3) does not exist as its padded area
	//printf("Traversing from col: %zu to %zu row:%zu\n", dx_lo, dx_hi, dy); 
    for(size_t dx = dx_lo + tid; dx <= dx_hi; dx += blockDim.x) {
      int16_t cellScore = (refSeq[dx]==qrySeq[(dy-dx)]) ? -penMatch : -penMisMatch;
      //printf("Comparing %c with %c, %hd match!\n", refSeq[dx], qrySeq[dy-dx], cellScore);
      cellScore += H[(dy-2)*refLen+dx-1];

      // update E,F,H
      E[dy*refLen+dx] = std::max(E[(dy-1)*refLen+dx-1]-penDelExt, H[(dy-1)*refLen+dx-1]-penDelOpen);
      F[dy*refLen+dx] = std::max(F[(dy-1)*refLen+dx  ]-penInsExt, H[(dy-1)*refLen+dx  ]-penInsOpen);
      //printf("Scores are: E:%d F:%d cellScore:%d \n", E[dy*refLen+dx], F[dy*refLen+dx], cellScore);
      H[dy*refLen+dx] = std::max(std::max(cellScore, (int16_t)0), std::max(E[dy*refLen+dx], F[dy*refLen+dx]));

      if (H[dy*refLen+dx] > (int16_t)thread_best_cell.score) {
        thread_best_cell.row = (uint16_t)dy;
        thread_best_cell.col = (uint16_t)dx;
        thread_best_cell.score = (uint16_t)H[dy*refLen+dx];
	//printf("New max score: %d at (%d, %d) aka (%d, %d)\n", thread_best_cell.score, thread_best_cell.col, thread_best_cell.row, thread_best_cell.col, thread_best_cell.row-thread_best_cell.col);
      }
    }
    __syncthreads();
  }

  // Read input from front buffer to shared
  shared_best[tid] = thread_best_cell;
  __syncthreads();
  //printf("New max score: %d at (%d, %d) aka (%d, %d)\n", shared_best[0].score, shared_best[0].col, shared_best[0].row, shared_best[0].col, shared_best[0].row-shared_best[0].col);

  // Shared reduction
  for (uint32_t i = 1; i < blockDim.x; i *= 2)
  {
      if (tid % (2 * i) == 0
           && tid + i < blockDim.x // this is an optional check so we can launch with thread count not a power of 2
         ) {
	if(shared_best[tid].score < shared_best[tid+i].score) {
	  shared_best[tid] = shared_best[tid+i];
	}
      }
      __syncthreads();
  }

  // Write result from shared to back buffer
  if (tid == 0) {
        *best_cell = shared_best[tid];
	//printf("HIP Final: %d at (%d, %d) aka (%d, %d)\n", best_cell->score, best_cell->col, best_cell->row, best_cell->col, best_cell->row-best_cell->col);
  }

}

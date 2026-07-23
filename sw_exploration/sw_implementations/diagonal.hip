#include "swag.h"
/* Diagonally prealigned
 * Changes the algorithm in memory accesses, but it is essentially the same.
 */

#include <hip/hip_runtime.h>

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
	//printf("New max score: %d at (%d, %d) aka (%d, %d)\n", best_cell->score, best_cell->col, best_cell->row, best_cell->col, best_cell->row-best_cell->col);
  }

}

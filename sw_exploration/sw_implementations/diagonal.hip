#include "diagonal.h"
#include <algorithm>

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

// Internal-only to alignBatch: row must hold the raw antidiagonal index (dy)
// Although rowslengths and columns are not to go over 2**16, even a relatively 
// short align query length of 256 can run out of 32 bits after double digit 
// million amount:
// rows=math.log(1e8 * 256 + 256,2)=34.5754247590989
// (approximate bits necessary for 10 million aligns of 256x256)
typedef struct bestCellTall {
  uint64_t row;
  uint16_t col;
  uint16_t score;
} bestCellTall;

// Internal-only to alignBatch: upper bound on how many alignments can be
// simultaneously "in flight" within a single antidiagonal row of the
// interleaved sweep. Since most aligns usually have a similar to 3x difference 
// in length between query and reference lengths for short aligns, this number should 
// most likely be in between 2 and 4. But if reference length is really much higher
// that the query length, this could be exceeded.
#define MAX_INFLIGHT_ALIGNS 8

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

  //int g_idx = threadIdx.x + blockIdx.x * blockDim.x;

  // we have numAligns number of alignments, we divide evenly with blocks
  const int ALIGNS_PER_BLOCK = (numAligns + gridDim.x - 1) / gridDim.x;
  int tid = threadIdx.x;

  // we're going to treat refLens and qryLens as if their prefixed padding is part of the sequence
  uint16_t refLen = max_refLen+1;
  uint16_t qryLen = max_qryLen+1;

  // We must keep our offset of our refIndex at all times
  // This alignIndex records which align we're at. So if we're working on the 100th alignment pair
  // This would be 100. Advances when it finishes the alignment.
  const int alignIndexBase = blockIdx.x * ALIGNS_PER_BLOCK; // this block's first (global) alignment index
  int alignIndex = alignIndexBase;
  const size_t TOTAL_DP_ROWS = refLen + (numAligns * qryLen) - 1;

  // Threads should be aligned as length of reference
  if (tid > refLen) {
      printf("Thread ID %d beyond reflen: %d", tid, refLen);
      return;
  };

  // How many alignments can share a single antidiagonal row: a row spans up
  // to refLen-1 columns, so qryInd can cross floor((refLen-1)/qryLen)
  // alignment boundaries; +1 for the alignment it started in, +1 more for
  // margin. Clamped to MAX_INFLIGHT_ALIGNS
  // e.g: Say 5 aligns, each 4x13 (including padding in lengths)
  //      (# is the alignment the cell is for)
  //                  0
  //                  0 0
  //                  0 0 0
  //                  0 0 0 0
  //                 -1-0 0 0 0
  //                 |1|1 0 0 0 0
  //                 |1|1 1 0 0 0 0
  //                 |1|1 1 1 0 0 0 0
  //                 |2|1 1 1 1 0 0 0 0
  //                 |2|2 1 1 1 1 0 0 0 0
  //                 |2|2 2 1 1 1 1 0 0 0 0
  // 0th alignment   |2|2 2 2 1 1 1 1 0 0 0 0
  // done, perform   |3|2 2 2 2 1 1 1 1 0 0 0 0
  // reduction and   |3|3 2 2 2 2 1 1 1 1 0 0 0 
  //   record best   |3|3 3 2 2 2 2 1 1 1 1 0 0
  //         score ->|3|3 3 3 2 2 2 2 1 1 1 1 0
  //                 |4|3 3 3 3 2 2 2 2 1 1 1 1
  // inflight=4 as ->-4-4 3 3 3 3 2 2 2 2 1 1 1
  // a single thread  4 4 4 3 3 3 3 2 2 2 2 1 1
  // is keeping track 4 4 4 4 3 3 3 3 2 2 2 2 1
  // of las 4 scores    4 4 4 4 3 3 3 3 2 2 2 2
  // until next align is  4 4 4 4 3 3 3 3 2 2 2
  // ready for reduction    4 4 4 4 3 3 3 3 2 2
  //                          4 4 4 4 3 3 3 3 2
  //                            4 4 4 4 3 3 3 3
  //                              4 4 4 4 3 3 3
  //                                4 4 4 4 3 3
  //                                  4 4 4 4 3
  //                                    4 4 4 4
  //                                      4 4 4
  //                                        4 4
  //                                          4

  const int inflight = min((int)((refLen - 1) / qryLen + 2), MAX_INFLIGHT_ALIGNS);
  if (tid == 0 && blockIdx.x == 0 && (int)((refLen - 1) / qryLen + 2) > MAX_INFLIGHT_ALIGNS) {
    printf("alignBatch: refLen/qryLen ratio needs %d in-flight slots, capped at %d -- results may be wrong\n",
           (int)((refLen - 1) / qryLen + 2), MAX_INFLIGHT_ALIGNS);
  }

  // Space to store best_cell for reduction after every thread calculates a thread-local best cell
  extern __shared__ char smem[];

  // For this reason shared_best is our reduction space for work.
  bestCellTall* shared_best = reinterpret_cast<bestCellTall*>(smem);
  // Buffer used to copy the best of every align until the end of the kernel.
  // Sized (by the host, see alignBatchNpar) and indexed for just THIS
  // block's own ALIGNS_PER_BLOCK share, not the whole batch -- indices are
  // alignIndex - alignIndexBase, i.e. block-relative. Sizing it for the
  // whole numAligns (as before) blew past the device's per-block shared
  // memory limit for large batches; each block only ever writes its own
  // range anyway, so it only needs to hold that much.
  bestCell* final_best = reinterpret_cast<bestCell*>(smem + (blockDim.x * sizeof(struct bestCellTall)));

  // Each thread needs a way to keep track of `inflight` alignments at once,
  // since that many can be simultaneously active within one row.
  bestCellTall thread_best_cell[MAX_INFLIGHT_ALIGNS];
  for (int i = 0; i < MAX_INFLIGHT_ALIGNS; i++) thread_best_cell[i] = {0, 0, 0};

  const int8_t penMatch     = penalties.match;
  const int8_t penMisMatch  = penalties.mismatch;
  const int8_t penDelOpen   = penalties.delOpen;
  const int8_t penDelExt    = penalties.delExt;
  const int8_t penInsOpen   = penalties.insOpen;
  const int8_t penInsExt    = penalties.insExt;

  
  // GROUP_START/GROUP_END are THIS block's own alignment-index boundaries
  // (in qryInd space), not the whole batch's. Each block must ramp up at its
  // own start and ramp down at its own end so it never reads/writes a cell
  // that belongs to a neighboring block. This ensures overlaps between blocks
  // don't affect each other.
  const size_t GROUP_START = (size_t)alignIndex * qryLen;
  const size_t GROUP_END = (size_t)(alignIndex + ALIGNS_PER_BLOCK) * qryLen;
  const size_t ROW_END = std::min<size_t>(GROUP_END + refLen, TOTAL_DP_ROWS);

  // GROUP_START ->                    
  //                  *                    
  // dy starts   ->   * *                  
  //      ↓           * * *                
  //  moves down      * * * *     dy marks right
  //                  * * * * * ⬐ bound in a row
  //                  * * * * * *          
  //                  * * * * * * *        
  //                  * * * * * * * *      
  //                  * * * * * * * * *    
  //                  * * * * * * * * * *  
  //                  * * * * * * * * * * *
  //                  * * * * * * * * * * * *
  //                  * * * * * * * * * * * * *
  //                  * * * * * * * * * * * * *
  //                  * * * * * * * * * * * * *
  //                  *.......................*
  //                  *...multiple alignments * 
  //                  *...happen for a while .* 
  //                  *.......................*
  //                  * * * * * * * * * * * * *
  //                  * * * * * * * * * * * * *
  // GROUP_END ->     * * * * * * * * * * * * *
  //                    * * * * * * * * * * * *
  //                      * * * * * * * * * * *
  //        dx marks left ⬏ * * * * * * * * * *
  //       bound in a row     * * * * * * * * *
  //                            * * * * * * * *
  //                              * * * * * * *
  //                                * * * * * *
  //                                  * * * * *
  //                                    * * * *
  //                                      * * *
  //                                        * *
  // ROW_END ->                               *

  // Start 2 rows past GR UP_START: the smallest dy with a real (non-padding)
  // cell for this group is dx=1, qryInd=GROUP_START+1, i.e. dy=GROUP_START+2.
  // Rows GROUP_START/GROUP_START+1 are boundary/padding cells that are either
  // owned by the previous block or fall out of range on both sides -- either
  // way this block must not touch them (also avoids dy-GROUP_START underflow
  // in dx_hi below, since dy is unsigned).
  for(size_t dy = GROUP_START + 2; dy < ROW_END; dy++) {

    // Bounds for the initial and final paddings, now relative to THIS
    // block's own group instead of the whole batch.
    size_t dx_lo = (dy >= GROUP_END) ? (dy - GROUP_END + 1) : 1;
    size_t dx_hi = (dy < GROUP_START + refLen) ? (dy - GROUP_START - 1) : (refLen - 1);
    // dx_lo is never 0 because 0th column is just padding and dependency addressing breaks

    // Control variable
    size_t qryInd = dy - dx_lo; // The index of our dy character in query
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
        // alignment Double Buffer Index, into this thread's own local slots
        int alignDBI = curAlignIndex % inflight;

        if (H[dy*refLen+dx] > (int16_t)thread_best_cell[alignDBI].score) {
          thread_best_cell[alignDBI].row = (uint64_t)dy;
          thread_best_cell[alignDBI].col = (uint16_t)dx;
          thread_best_cell[alignDBI].score = (uint16_t)H[dy*refLen+dx];
          //printf("New thread max score: %d at (%d, %d) aka (%d, %d) alignDBI:%d\n", thread_best_cell[alignDBI].score, thread_best_cell[alignDBI].col, thread_best_cell[alignDBI].row, thread_best_cell[alignDBI].col, thread_best_cell[alignDBI].row-thread_best_cell[alignDBI].col, alignDBI);
        }
      }
    }

    // every thread must finish writing to the row
    __syncthreads();

    // check if we're at a reduction point because an alignment is done
    // So we should perform reduction. This periodicity is relative to
    // GROUP_START, not the raw dy -- otherwise a block starting partway
    // through the batch sees the trigger fire at a phase that belongs to
    // an alignment owned by an earlier block, and mislabels it as its own.
    size_t localDy = dy - GROUP_START;
    if(localDy > (refLen-1) &&  // The first qryLen rows are kind of like an offset. Sorry this isn't a place to be explaining it.
	( (localDy+1) - (refLen-1) ) % qryLen == 0) { // give a qryLen offSet to dy because we finish an align every refLen after qryLen rows
      // Which double-buffer slot holds the alignment that just completed.
      // `alignIndex` (this block's tracked GLOBAL alignment index, not yet
      // incremented) is authoritative -- re-deriving the parity from dy via
      // the periodicity formula implicitly assumed GROUP_START==0 and broke
      // for blocks whose first owned alignment has an odd global index.
      int alignDBI = alignIndex % inflight;
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
      // save the best thread into shared space for best alignment per pair + fix absolute index -> relative index
      if(tid == 0) {
        int localAlignIndex = alignIndex - alignIndexBase; // final_best is block-relative
        final_best[localAlignIndex].score = shared_best[0].score;
        final_best[localAlignIndex].col   = shared_best[0].col;
        final_best[localAlignIndex].row   = shared_best[0].row - (alignIndex*qryLen) - shared_best[0].col;
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


  // Write results from shared to back main buffer.
  // Only write the slice of alignments THIS block actually owns -- looping
  // over all numAligns here would let every block race to overwrite every
  // other block's results in the global best_cells array. final_best itself
  // is now block-relative sized/indexed (see above), so index it by
  // i - alignStart to read back the same slice.
  int alignStart = blockIdx.x * ALIGNS_PER_BLOCK;
  int alignStop = min((int)(alignStart + ALIGNS_PER_BLOCK), (int)numAligns);
  if (tid == 0) {
    for(int i = alignStart; i < alignStop; i++) {
      best_cells[i] = final_best[i - alignStart];
      //printf("HIP Final %d: %d at (%d, %d) aka (%d, %d)\n", i, best_cells[i].score, best_cells[i].col, best_cells[i].row, best_cells[i].col, (best_cells[i].row-i*qryLen)-best_cells[i].col);
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

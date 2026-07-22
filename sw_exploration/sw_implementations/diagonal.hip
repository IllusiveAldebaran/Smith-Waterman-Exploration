/* Diagonally prealigned
 * Changes the algorithm in memory accesses, but it is essentially the same.
 */
void alignOne(const uint16_t refLen, const uint16_t qryLen,
			 const int8_t penalties[6],
			 const char* refSeq, const char* qrySeq,
			 int16_t* H, int16_t* E, int16_t* F,
			 bestCell* best_cell,
			 float* floatCounters, int nfC,
			 int* intCounters, int niC) {
 
  const int8_t penMatch     = penalties[0];
  const int8_t penMisMatch  = penalties[1];
  const int8_t penDelOpen   = penalties[2];
  const int8_t penDelExt    = penalties[3];
  const int8_t penInsOpen   = penalties[4];
  const int8_t penInsExt    = penalties[5];


  //printf("horizontals: %d\n", refLen+qryLen-1);
  for(size_t dy = 2; dy < refLen + qryLen - 1; dy++) {
    size_t dx_lo = (dy >= qryLen) ? (dy - qryLen + 1) : 1;
    size_t dx_hi = (dy < refLen) ? (dy - 1) : (refLen - 1);

    // iterate through all cols but select to do run the alignment algorithm or not.
    // Makes vectorization (and possibly later on batched alignment) possible
    // check beyond diagonal.
    // Example in (dy, dx) (2, 3) does not exist as its padded area
	//printf("Traversing from col: %zu to %zu row:%zu\n", dx_lo, dx_hi, dy); 
    for(size_t dx = dx_lo; dx <= dx_hi; dx++) {
      int16_t cellScore = (refSeq[dx]==qrySeq[(dy-dx)]) ? -penMatch : -penMisMatch;
      //printf("Comparing %c with %c, %hd match!\n", refSeq[dx], qrySeq[dy-dx], cellScore);
      cellScore += H[(dy-2)*refLen+dx-1];

      // update E,F,H
      E[dy*refLen+dx] = std::max(E[(dy-1)*refLen+dx-1]-penDelExt, H[(dy-1)*refLen+dx-1]-penDelOpen);
      F[dy*refLen+dx] = std::max(F[(dy-1)*refLen+dx  ]-penInsExt, H[(dy-1)*refLen+dx  ]-penInsOpen);
      //printf("Scores are: E:%d F:%d cellScore:%d \n", E[dy*refLen+dx], F[dy*refLen+dx], cellScore);
      H[dy*refLen+dx] = std::max(std::max(cellScore, (int16_t)0), std::max(E[dy*refLen+dx], F[dy*refLen+dx]));

      if (H[dy*refLen+dx] > (int16_t)best_cell->score) {
        best_cell->row = (uint16_t)dy;
        best_cell->col = (uint16_t)dx;
        best_cell->score = (uint16_t)H[dy*refLen+dx];
      }
    }
  }

}

#include <stdio.h>
#include <string.h>
#include <cstdlib>

#include "diagonal.c"

#include "seq_utils.h"

#include "kseq++/seqio.hpp"
#include <unistd.h>

using namespace klibpp;

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
    printf("Usage: ./dp_test <reference.fa> <query.fa> <threads>\n");
    printf("Threads should be <min(reference length, hardware limit)\n");
    return 1;
  }

  size_t i = 0;
  // get an array of all lengths
  size_t* refLens;
  size_t* qryLens;
  char* refContiguous;
  char* qryContiguous;
  size_t refLenC; // sequence lengths of contiguous batched sequence
  size_t qryLenC;
  // Number of sequences
  size_t refNum;
  size_t qryNum;

  // Keeping track of maximum length of qry and len... but the assumption is that they're all the same though
  size_t max_refLen;
  size_t max_qryLen;

  {
    size_t i = 0;

    //printf("Testing to read the sequences\n");
    SeqStreamIn ref_iss(argv[1]);
    auto refVec = ref_iss.read();
    // get an array of all lengths
    refLens = (size_t*)calloc(refVec.size(),sizeof(size_t));
    max_refLen = refVec[0].seq.size();
    refNum = refVec.size();
    // loop once to find the maximum sequence, fill up refLens too
    for (auto & element : refVec) {
      size_t seqLen = element.seq.size();
      if(max_refLen != seqLen) {
	printf("Warning! Different sizes reference sequences (sequence %ld of size: %ld)\n", i, seqLen);
	max_refLen = (seqLen > max_refLen) ? seqLen : max_refLen;
      }

      refLens[i] = seqLen;
      i++;
    }
    // now that we know the maximum we use this to allocate and then fill up our batched sequence
    // Since we pad the sequence and a prefixed padding, we do +1 for every sequence and +1 for the start
    refContiguous = (char*)calloc((refNum*(max_refLen+1)+1),sizeof(char));
    i = 0;
    for (auto & element : refVec) {
      size_t seqLen = element.seq.size();
      memcpy((refContiguous+1)+i, element.seq.c_str(), seqLen);
      i += max_refLen+1;
    }


    SeqStreamIn qry_iss(argv[2]);
    auto qryVec = qry_iss.read();
    qryLens = (size_t*)(malloc(qryVec.size()*sizeof(size_t)));
    max_qryLen = qryVec[0].seq.size();
    qryNum = qryVec.size();
    // loop once to find the maximum sequence, fill up qryLens too
    i = 0;
    for (auto & element : qryVec) {
      size_t seqLen = element.seq.size();
      if(max_qryLen != seqLen) {
	printf("Warning! Different sizes query sequences (sequence %ld of size: %ld)\n", i, seqLen);
	max_qryLen = (seqLen > max_qryLen) ? seqLen : max_qryLen;
      }

      qryLens[i] = seqLen;
      i++;
    }
    // now that we know the maximum we use this to allocate and then fill up our batched sequence
    // Since we pad the sequence and a prefixed padding, we do +1 for every sequence and +1 for the start
    qryContiguous = (char*)calloc((qryVec.size()*(max_qryLen+1)+1),sizeof(char));
    i = 0;
    for (auto & element : qryVec) {
      size_t seqLen = element.seq.size();
      memcpy((qryContiguous+1)+i, element.seq.c_str(), seqLen);
      i += max_qryLen+1;
    }
  }

  // we are including the prefixed padding we are going to add
  //uint16_t refLen = strlen(argv[1])+1;
  //uint16_t qryLen = strlen(argv[2])+1;
  int npar = atoi(argv[3]);


  // assumption is that qryNum = refNum... because it should. User input error elsewise.
  uint32_t numAligns = qryNum; // don't expect to be doing more than 4.29 billion aligns (max uint32_t)
  // qryNum should also just be equal to refNum because we are alignins them to each other
  struct bestCell* best_cells = (struct bestCell*)calloc(numAligns, sizeof(struct bestCell));

  // + 1 because of 0th prefixed padding
  refLenC = numAligns * (max_refLen+1);
  qryLenC = numAligns * (max_qryLen+1);

  // H/E/F are GPU-only scratch now, allocated/freed internally by
  // alignBatchNpar -- never passed in or copied back to the host.
  Penalties penalties = {MATCH, MISMATCH, PENDELO, PENDELE, PENINSO, PENINSE};
  // just temporary empty variables... for prof/counting
  const int nfC = 0;
  const int niC = 0;
  float fCount[nfC];
  int intCount[niC];

  alignBatchNpar(max_refLen, max_qryLen,
	   	penalties,
	   	refContiguous, qryContiguous,
	   	best_cells,
	   	numAligns,
	   	npar,
      	   	fCount, nfC, intCount, niC);
  // fills best_cell


  for(i = 0; i < refNum; i++){
    //printf("%c", (refContiguous[i] == 0 ? '0' : refContiguous[i] ));// could be accidentally putting null characters btw
    //printf("Best Cell: (%d, %d) diagonal aka (%d, %zu) score: %d\n", best_cells[i].col, best_cells[i].row, best_cells[i].col, (best_cells[i].row - i*(max_refLen+1))-best_cells[i].col, best_cells[i].score);
    printf("Best Cell: (%d, %d) score: %d\n", best_cells[i].col, best_cells[i].row, best_cells[i].score);
    // print out part of the DP related to this alignment
    int qryOff = i * (max_qryLen+1); // offsets
    int refOff = i * (max_refLen+1);
    //showDP((uint8_t*)(refContiguous+refOff), max_refLen+1, (uint8_t*)(qryContiguous+qryOff), qryLens[i]+1, H+((max_refLen+1)*(max_qryLen+1))*i, true);
  }

  // print DP to stdout, not accurate info on the reference sequence but close enough
  //showDP((uint8_t*)(refContiguous), max_refLen+1, (uint8_t*)(qryContiguous), qryLenC, H, true);

  free(best_cells);


  //printf("Putting refContiguous of length %ld\n", refNum*max_refLen+1);
  //for(i = 0; i < refNum*(max_refLen+1)+1; i++){
  //  printf("%c", (refContiguous[i] == 0 ? '0' : refContiguous[i] ));// could be accidentally putting null characters btw
  //}
  //printf("\nPutting qrySequence of length %ld\n", qryNum*max_qryLen+1);
  //for(i = 0; i < qryNum*(max_qryLen+1)+1; i++){
  //  printf("%c", (qryContiguous[i] == 0 ? '0' : qryContiguous[i] ));// could be accidentally putting null characters btw
  //}

  printf("\n");
  free(refLens);
  free(qryLens);
  free(refContiguous);
  free(qryContiguous);

  return 0;
}

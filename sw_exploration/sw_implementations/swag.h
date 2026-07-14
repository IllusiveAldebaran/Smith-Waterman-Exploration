#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>

void alignOne(const uint16_t refLen, const uint16_t qryLen, const int8_t penalties[6], const char* refSeq, const char* qrySeq, int16_t* H);

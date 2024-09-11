#pragma once

#include "common.h"

namespace StreamCompaction {
    namespace Naive {
        StreamCompaction::Common::PerformanceTimer& timer();
        void scan(int n, int *odata, const int *idata);
        __global__ void naiveScan(int n, int depth, const int* inputBuf, int* outputBuf);
        __global__ void shiftRight(int n, const int* inputBuf, int* outputBuf);
    }
}

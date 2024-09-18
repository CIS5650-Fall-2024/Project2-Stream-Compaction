#pragma once

#include "common.h"

namespace StreamCompaction {
    namespace Efficient {
        StreamCompaction::Common::PerformanceTimer& timer();

        void scan(int n, int *odata, const int *idata);
        void scan(int n_padded, int* dev_data, int* stored_sums, int offset);
        __global__ void kernScan(int n, int numLevels, int* dev_data, int* stored_sums);
        __global__ void kernIncrement(int n, int* dev_data, int* sum_data);
        int compact(int n, int *odata, const int *idata);
    }
}

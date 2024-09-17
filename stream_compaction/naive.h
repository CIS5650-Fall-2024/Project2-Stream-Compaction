#pragma once

#include "common.h"

namespace StreamCompaction {
    namespace Naive {
        StreamCompaction::Common::PerformanceTimer& timer();

        __global__ void naive_scan_block(int n, int *data, int *block_sums = nullptr);
        void scan(int n, int *odata, const int *idata, int block_size = 128);
    }
}

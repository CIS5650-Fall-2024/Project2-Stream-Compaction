#pragma once

#include "common.h"

namespace StreamCompaction {
    namespace Efficient {
        StreamCompaction::Common::PerformanceTimer& timer();

        void scan(int n, int *odata, const int *idata);
        __global__ void kernUpSweep(int n, int depth, int* dev_data);
        __global__ void kernDownSweep(int n, int depth, int* dev_data);
        int compact(int n, int *odata, const int *idata);
    }
}

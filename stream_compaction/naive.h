#pragma once

#include "common.h"

namespace StreamCompaction {
    namespace Naive {
        StreamCompaction::Common::PerformanceTimer& timer();

        void scan(int n, int *odata, const int *idata);

        __global__ void scan_single_aggregate(int n, int *odata, const int *idata, int offset);

        __global__ void inclusive_to_exclusive(int n, int *incl, int *excl);

        __global__ void exclusive_to_inclusive(int n, int *excl, int *incl);
    }
}

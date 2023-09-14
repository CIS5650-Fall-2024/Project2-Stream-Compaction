#pragma once

#include "common.h"

namespace StreamCompaction {
    namespace Efficient {
        StreamCompaction::Common::PerformanceTimer& timer();
        void scanCore(int n, int* odata, const int* idata);
        void scan(int n, int *odata, const int *idata);

        int compact(int n, int *odata, const int *idata);
        __global__ void kernUpSweep(int n,int offset, int* odata1);
        __global__ void kernDownSweep(int n, int offset, int* odata1);
    }
}
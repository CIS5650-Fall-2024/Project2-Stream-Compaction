#pragma once

#include "common.h"

namespace StreamCompaction {
    namespace Efficient {
        StreamCompaction::Common::PerformanceTimer& timer();

        void scan(int n, int *odata, const int *idata, bool timeScan = true);

        int compact(int n, int *odata, const int *idata);

        void radixSort(int n, int* odata, const int* idata);
    }
}

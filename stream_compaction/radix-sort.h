#pragma once

#include "common.h"

namespace StreamCompaction {
    namespace RadixSort {
        StreamCompaction::Common::PerformanceTimer& timer();

        void sort(int n, int *odata, const int *idata);
        void sort(int n, int maxVal, int *odata, const int *idata);
    }
}

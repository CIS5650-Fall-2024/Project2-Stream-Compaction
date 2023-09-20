#pragma once

#include "common.h"

namespace StreamCompaction {
    namespace RadixSort {
        StreamCompaction::Common::PerformanceTimer& timer();

        void sort(int n, int* odata, const int* idata, int numBits);
        void sortShared(int n, int* odata, const int* idata, int numBits);
    }
}

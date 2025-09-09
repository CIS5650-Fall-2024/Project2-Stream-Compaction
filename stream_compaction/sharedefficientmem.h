#pragma once

#include "common.h"

namespace StreamCompaction {
    namespace SharedEfficientMem {

        StreamCompaction::Common::PerformanceTimer& timer();

        void scanSharedEfficient(int n, int* odata, const int* idata);

        int compactEfficient(int n, int* odata, const int* idata);
    }
}

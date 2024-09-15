#pragma once

#include "common.h"

namespace StreamCompaction {
    namespace Radix {
        StreamCompaction::Common::PerformanceTimer& timer();

        int getMaxBits(int n, const int *idata);

        void sort(int n, int *odata, const int *idata, int maxBits);
    }
}

#pragma once

#include "common.h"

namespace StreamCompaction {
    namespace Sort {
        StreamCompaction::Common::PerformanceTimer& timer();

        void radix_sort_thrust(int n, int *odata, const int *idata);

        void radix_sort(int n, int *odata, const int *idata);
    }
}
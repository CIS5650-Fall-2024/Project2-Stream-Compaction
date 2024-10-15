#pragma once

#include "common.h"

namespace StreamCompaction {
    namespace EfficientShare {
        StreamCompaction::Common::PerformanceTimer& timer();

        void scan(int n, int* odata, const int* idata);

        void scan_dev(int n, int* dev_odata);

        int compact(int n, int* odata, const int* idata);
    }
}

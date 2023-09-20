#pragma once

#include "common.h"

namespace StreamCompaction {
    namespace Efficient {
        StreamCompaction::Common::PerformanceTimer& timer();

        void scan(int n, int *odata, const int *idata);

        void _scan_dev(int n, int *dev_idata);

        int compact(int n, int *odata, const int *idata);
    }
}

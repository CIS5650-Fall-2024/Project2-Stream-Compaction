#pragma once

#include "common.h"

namespace StreamCompaction {
    namespace Efficient {
        StreamCompaction::Common::PerformanceTimer& timer();

        void scan(int n, int *odata, const int *idata);
        void scanShared(int n, int* datao, const int* datai);

        int compact(int n, int *odata, const int *idata);
    }
}

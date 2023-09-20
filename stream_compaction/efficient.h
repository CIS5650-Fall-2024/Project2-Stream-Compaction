#pragma once

#include "common.h"

namespace StreamCompaction {
    namespace Efficient {
        StreamCompaction::Common::PerformanceTimer& timer();

        void scan(int n, int *odata, const int *idata, int BLOCKSIZE = BLOCK_SIZE);

        int compact(int n, int *odata, const int *idata, int BLOCKSIZE = BLOCK_SIZE);
    }
}

#pragma once

#include "common.h"

#define MAX_BLOCK_SIZE 512
#define NUM_BANKS 32
#define LOG_NUM_BANKS 5
#define ZERO_BANK_CONFLICTS 0

#if ZERO_BANK_CONFLICTS
#define CONFLICT_FREE_OFFSET(n) \
	((n) >> NUM_BANKS + (n) >> (2 * LOG_NUM_BANKS))
#else
#define CONFLICT_FREE_OFFSET(n) ((n) >> LOG_NUM_BANKS)
#endif

namespace StreamCompaction {
    namespace EfficientOptimized {
        StreamCompaction::Common::PerformanceTimer& timer();

        void scan(int n, int* odata, const int* idata, bool startTimer = true, bool isHost = true);

        int compact(int n, int *odata, const int *idata);
    }
}

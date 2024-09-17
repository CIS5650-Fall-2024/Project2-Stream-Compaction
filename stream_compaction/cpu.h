#pragma once

#include "common.h"

namespace StreamCompaction {
    namespace CPU {
        StreamCompaction::Common::PerformanceTimer& timer();

        // Specify simulateGPUScan to choose between scanExclusiveSerial and 
        // scanExclusiveSimulateGPU.
        void scan(int n, int *odata, const int *idata, bool simulateGPUScan = true);

        // Completely serial version of exclusive scan in CPU.
        void scanExclusiveSerial(int n, int *odata, const int *idata);

        // CPU version of parallel algorithm. Mimics the GPU parallel algorithm in 
        // StreamCompaction::Naive::scan to some extent.
        void scanExclusiveSimulateGPU(int n, int *odata, const int *idata);

        int compactWithoutScan(int n, int *odata, const int *idata);

        int compactWithScan(int n, int *odata, const int *idata);
    }
}

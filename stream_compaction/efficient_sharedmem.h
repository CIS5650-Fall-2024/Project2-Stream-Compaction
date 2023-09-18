#pragma once

#include "common.h"
#include <vector>
namespace StreamCompaction {
    namespace EfficientSharedMem {
        StreamCompaction::Common::PerformanceTimer& timer();
        //RAII device buffer
        class gpuScanTempBuffer
        {
        public:
            gpuScanTempBuffer(int n, int block_array_size, const int* idata);
            ~gpuScanTempBuffer();
            std::vector<std::pair<int*,int*>> buffers;
            std::vector<int> numBlocks;
            std::vector<int> blockSizes;
            std::vector<int> sharedMemSize;
            std::vector<int> numWorkloads;
            int block_array_size;
        };

        void gpuScanWorkEfficientOptimized(const gpuScanTempBuffer& tmpBuf);
        void scan(int n, int *odata, const int *idata);
        int compact(int n, int *odata, const int *idata);
    }
}

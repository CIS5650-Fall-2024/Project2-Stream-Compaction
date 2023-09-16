#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

namespace StreamCompaction {
    namespace Efficient {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        __global__ void kernUpSweep(int* data, int n, int stride) {
            int index = blockIdx.x * blockDim.x + threadIdx.x + 1;
            if (index > n)return;
            int real_i = index * stride * 2 - 1;
            data[real_i] += data[real_i - stride];
        }

        __global__ void kernDownSweep(int* data, int n, int stride) {
            int index = blockIdx.x * blockDim.x + threadIdx.x + 1;
            if (index > n)return;
            int real_i = index * stride * 2 - 1;
            int t = data[real_i];
            data[real_i] += data[real_i - stride];
            data[real_i - stride] = t;
        }
        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int* odata, const int* idata) {
            int dMax = ilog2ceil(n), strideMax = 1 << (dMax - 1);
            int extended_n = 1 << dMax;
            int* dev_data;
            cudaMalloc((void**)&dev_data, sizeof(int) * extended_n);
            cudaMemset(dev_data, 0, sizeof(int) * extended_n);
            cudaMemcpy(dev_data, idata, sizeof(int) * n, cudaMemcpyHostToDevice);
            timer().startGpuTimer();
            dim3 fullBlocksPerGrid;
            // TODO
            for (int i = 1, int n = strideMax; i < strideMax; i <<= 1, n >>= 1)
            {
                fullBlocksPerGrid = ((n + blockSize - 1) / blockSize);
                kernUpSweep << <fullBlocksPerGrid, blockSize >> > (dev_data, n, i);
            }
            cudaMemset(&dev_data[extended_n - 1], 0, sizeof(int));
            for (int i = strideMax, int n = 1; i >= 1; i >>= 1, n <<= 1)
            {
                fullBlocksPerGrid = ((n + blockSize - 1) / blockSize);
                kernDownSweep << <fullBlocksPerGrid, blockSize >> > (dev_data, n, i);
            }
            timer().endGpuTimer();
            cudaMemcpy(odata, dev_data, sizeof(int) * n, cudaMemcpyDeviceToHost);
            cudaFree(dev_data);
        }

        /**
         * Performs stream compaction on idata, storing the result into odata.
         * All zeroes are discarded.
         *
         * @param n      The number of elements in idata.
         * @param odata  The array into which to store elements.
         * @param idata  The array of elements to compact.
         * @returns      The number of elements remaining after compaction.
         */
        int compact(int n, int* odata, const int* idata) {
            timer().startGpuTimer();
            // TODO
            timer().endGpuTimer();
            return -1;
        }
    }
}

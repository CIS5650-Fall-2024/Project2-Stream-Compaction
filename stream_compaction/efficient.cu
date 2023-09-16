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

        __global__ void kernUpSweep(int* data, int n, int offset) {
            int index = blockIdx.x * blockDim.x + threadIdx.x;
            int offset2 = offset << 1;
            if (index + offset2 <= n && (index & (offset2 - 1)) == 0)
                data[index + offset2 - 1] += data[index + offset - 1];
        }

        __global__ void kernDownSweep(int* data, int n, int offset) {
            int index = blockIdx.x * blockDim.x + threadIdx.x;
            int offset2 = offset << 1;
            if (index + offset2 <= n && (index & (offset2 - 1)) == 0) {
                int t = data[index + offset2 - 1];
                data[index + offset2 - 1] += data[index + offset - 1];
                data[index + offset - 1] = t;
            }
        }
        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int* odata, const int* idata) {
            int dMax = ilog2ceil(n);
            int extended_n = 1 << dMax;
            int* dev_data;
            cudaMalloc((void**)&dev_data, sizeof(int) * extended_n);
            cudaMemset(dev_data, 0, sizeof(int) * extended_n);
            cudaMemcpy(dev_data, idata, sizeof(int) * n, cudaMemcpyHostToDevice);
            dim3 fullBlocksPerGrid((extended_n + blockSize - 1) / blockSize);
            timer().startGpuTimer();
            // TODO
            for (int i = 0; i < dMax - 1; i++)
            {
                kernUpSweep << <fullBlocksPerGrid, blockSize >> > (dev_data, extended_n, 1 << i);
            }
            cudaMemset(&dev_data[extended_n - 1], 0, sizeof(int));
            for (int i = dMax - 1; i >= 0; i--)
            {
                kernDownSweep << <fullBlocksPerGrid, blockSize >> > (dev_data, extended_n, 1 << i);
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

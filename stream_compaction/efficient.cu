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

        __global__ void kernUpSweep(int* data, int n, int d) {
            int index = blockIdx.x * blockDim.x + threadIdx.x;
            int shift = 1 << d, shift2 = shift << 1;
            if (index > n || index + shift2 > n || index + shift > n)
                return;
            if ((index & (shift2 - 1)) == 0)
                data[index + shift2 - 1] += data[index + shift - 1];
        }

        __global__ void kernDownSweep(int* data, int n, int d, bool isStart) {
            int index = blockIdx.x * blockDim.x + threadIdx.x;
            int shift = 1 << d, shift2 = shift << 1;
            if (isStart)
            {
                data[shift - 1] = 0;
                return;
            }
            if (index > n || index + shift2 > n || index + shift > n)
                return;
            if ((index & (shift2 - 1)) == 0) {
                int t = data[index + shift2 - 1];
                data[index + shift2 - 1] += data[index + shift - 1];
                data[index + shift - 1] = t;
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
            for (int i = 0; i < dMax; i++)
            {
                kernUpSweep << <fullBlocksPerGrid, blockSize >> > (dev_data, extended_n, i);
            }
            for (int i = dMax; i >= 0; i--)
            {
                kernDownSweep << <fullBlocksPerGrid, blockSize >> > (dev_data, extended_n, i, i == dMax);
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

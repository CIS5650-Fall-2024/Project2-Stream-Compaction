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

        __global__ void kernUpSweep(int n, int d, int* data) {
            int index = blockIdx.x * blockDim.x + threadIdx.x;
            if (index >= n) {
                return;
            }

            int offset1 = 1 << d;
            int offset2 = 1 << (1 + d);
            if (index % offset2 == 0) {
                data[index + offset2 - 1] +=
                    data[index + offset1 - 1];
            }
        }

        __global__ void kernDownSweep(int n, int d, int *data) {
            int index = blockIdx.x * blockDim.x + threadIdx.x;
            if (index >= n) {
                return;
            }

            int offset1 = 1 << d;
            int offset2 = 1 << (1 + d);
            if (index % offset2 == 0) {
                int left = data[index + offset1 - 1]; // Save left child
                data[index + offset1 - 1] = data[index + offset2 - 1]; // Set left child to this node’s value
                data[index + offset2 - 1] += left; // Set right child to old left value +
                                                   // this node’s value
            }
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            int* dev_buf;

            int power2 = ilog2ceil(n);
            int chunkSize = 1 << power2;

            dim3 blocksPerGrid((chunkSize + blockSize - 1) / blockSize);
            size_t arrSize = chunkSize * sizeof(int);

            cudaMalloc((void**)&dev_buf, arrSize);
            cudaMemcpy(dev_buf, idata, arrSize, cudaMemcpyHostToDevice);

            timer().startGpuTimer();
            // TODO
            // Up Sweep
            for (int d = 0; d <= power2-1; ++d) {
                kernUpSweep <<<blocksPerGrid, blockSize>>> (chunkSize, d, dev_buf);
            }
            
            // Down Sweep
            cudaMemset(dev_buf + chunkSize - 1, 0, sizeof(int)); // set root to zero
            for (int d = power2-1; d >= 0; --d) {
                kernDownSweep <<<blocksPerGrid, blockSize>>> (chunkSize, d, dev_buf);
            }
            timer().endGpuTimer();

            cudaMemcpy(odata, dev_buf, arrSize, cudaMemcpyDeviceToHost);
            cudaFree(dev_buf);
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
        int compact(int n, int *odata, const int *idata) {
            timer().startGpuTimer();
            // TODO
            timer().endGpuTimer();
            return -1;
        }
    }
}

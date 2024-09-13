#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"
#include <algorithm>

namespace StreamCompaction {
    namespace Efficient {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            int n_padded = pow(2, ilog2ceil(n)); // pad array to neared power of two

            int* dev_data;
            cudaMalloc((void**) &dev_data, n_padded * sizeof(int));
            cudaMemcpy(dev_data, idata, n * sizeof(int), cudaMemcpyHostToDevice);

            // Pad input data to be a power of two, if needed.
            if (n < n_padded) {
                cudaMemset(dev_data + n, 0, (n_padded - n) * sizeof(int));
            }

            timer().startGpuTimer();
            
            for (int depth = 0; depth < ilog2ceil(n_padded); ++depth) {
                int totalOperations = n_padded / (1 << depth);
                int blockSize = std::min(totalOperations, 1024); // 1024 is a hardware limitation
                dim3 blocksPerGrid = (totalOperations + blockSize - 1 / blockSize);

                kernUpSweep<<<blockSize, blocksPerGrid>>>(totalOperations, depth, dev_data);
                cudaDeviceSynchronize();
            }

            // Pre-step for downsweep
            cudaMemset(dev_data + n_padded - 1, 0, sizeof(int));

            for (int depth = ilog2ceil(n_padded) - 1; depth >= 0; --depth) {
                int totalOperations = n_padded / (1 << depth);
                int blockSize = std::min(totalOperations, 1024); // 1024 is a hardware limitation
                dim3 blocksPerGrid = (totalOperations + blockSize - 1 / blockSize);

                kernDownSweep<<<blockSize, blocksPerGrid>>>(totalOperations, depth, dev_data);
                cudaDeviceSynchronize();
            }

            timer().endGpuTimer();
            cudaMemcpy(odata, dev_data, n_padded * sizeof(int), cudaMemcpyDeviceToHost);
            cudaFree(dev_data);
        }


        /**
         * n is the total number of threads doing work on this iteration,
         * not necessarily the number of elements in the overall array.
         */
        __global__ void kernUpSweep(int n, int depth, int* dev_data) {
            int threadId = (blockDim.x * blockIdx.x) + threadIdx.x;
            if (threadId >= n) return;

            int twoToDepthPlusOne = (1 << (depth + 1));
            int twoToDepth = (1 << depth);
            int leftChildIdx = (threadId * twoToDepthPlusOne) + twoToDepth - 1;
            int rightChildIdx = (threadId * twoToDepthPlusOne) + twoToDepthPlusOne - 1;

            dev_data[rightChildIdx] += dev_data[leftChildIdx];
        }

        __global__ void kernDownSweep(int n, int depth, int* dev_data) {
            int threadId = (blockDim.x * blockIdx.x) + threadIdx.x;
            if (threadId >= n) return;

            int twoToDepthPlusOne = (1 << (depth + 1));
            int twoToDepth = (1 << depth);
            int leftChildIdx = (threadId * twoToDepthPlusOne) + twoToDepth - 1;
            int rightChildIdx = (threadId * twoToDepthPlusOne) + twoToDepthPlusOne - 1;

            int leftVal = dev_data[leftChildIdx];
            dev_data[leftChildIdx] = dev_data[rightChildIdx];
            dev_data[rightChildIdx] += leftVal;
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

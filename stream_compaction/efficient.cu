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


        __global__ void UpSweepAtDepthD(int n, int d, int* buffer) {
            int k = (blockIdx.x * blockDim.x) + threadIdx.x;
            int offsetBetweenMains = pow(2, d + 1);
            int actualOffset = pow(2, d);

            int index = (k + 1) * offsetBetweenMains - 1;
            if (index < n) {
                buffer[index] = buffer[index] + buffer[index - actualOffset];
            }
        }

        __global__ void DownSweepAtDepthD(int n, int d, int* buffer) {
            int k = (blockIdx.x * blockDim.x) + threadIdx.x;
            int offsetBetweenMains = pow(2, d + 1);
            int actualOffset = pow(2, d);

            int index = (k + 1) * offsetBetweenMains - 1;
            if (index < n) {
                //left child index
                int leftChildIndex = index - actualOffset;
                int rightChildIndex = index;
                int leftChildSave = buffer[leftChildIndex];
                buffer[leftChildIndex] = buffer[index];
                buffer[rightChildIndex] = leftChildSave + buffer[index];
            }
        }
        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            int* buffer;
            int size = pow(2, ilog2ceil(n));
            //int size = n;
            cudaMalloc((void**)&buffer, size * sizeof(int));
            cudaMemset(buffer, 0, size * sizeof(int));
            cudaMemcpy(buffer, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            timer().startGpuTimer();
            // TODO

            //UpSweep (parallel reduction)
            for (int d = 0; d < ilog2ceil(size); d++) {
                dim3 fullBlocksPerGrid((size + blockSize - 1) / blockSize);
                UpSweepAtDepthD<<<fullBlocksPerGrid, blockSize>>>(size, d, buffer);
                checkCUDAError("UpSweepAtDepthD failed!");
            }

            //DownSweep
            cudaMemset(buffer + (size - 1), 0, sizeof(int));
            for (int d = ilog2ceil(n) - 1; d >= 0; d--) {
                dim3 fullBlocksPerGrid((size + blockSize - 1) / blockSize);
                DownSweepAtDepthD<<<fullBlocksPerGrid, blockSize>>>(size, d, buffer);
                checkCUDAError("DownSweepAtDepthD failed!");
            }


            timer().endGpuTimer();
            //Cpy data back to CPU
            cudaMemcpy(odata, buffer, size * sizeof(int), cudaMemcpyDeviceToHost);
            cudaFree(buffer);
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

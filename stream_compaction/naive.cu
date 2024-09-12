#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"
#include <algorithm>
#include <iostream>

namespace StreamCompaction {
    namespace Naive {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }
        // TODO: __global__

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {        
            int nearestPowerOfTwo = pow(2, ilog2ceil(n));
            int* outputBuf;
            int* inputBuf;
            cudaMalloc((void**)&outputBuf, nearestPowerOfTwo * sizeof(int));
            cudaMalloc((void**)&inputBuf, nearestPowerOfTwo * sizeof(int));
            cudaMemcpy(inputBuf, idata, nearestPowerOfTwo * sizeof(int), cudaMemcpyHostToDevice);


            // Pad input data to be a power of two, if needed.
            if (n < nearestPowerOfTwo) {
                cudaMemset(inputBuf + n, 0, (nearestPowerOfTwo - n) * sizeof(int));
            }

            timer().startGpuTimer();

            for (int depth = 1; depth <= ilog2ceil(nearestPowerOfTwo); ++depth) {
                int blockSize = std::min(nearestPowerOfTwo, 1024); // cap at 1024 threads, hardware limitation.
                dim3 blocksPerGrid((nearestPowerOfTwo + blockSize - 1) / blockSize); // note integer division
                naiveScan<<<blocksPerGrid, blockSize>>>(nearestPowerOfTwo, depth, inputBuf, outputBuf);

                std::swap(outputBuf, inputBuf);
            }

            // Convert inclusive scan to exclusive scan
            int blockSize = std::min(n, 1024);
            dim3 blocksForShift((nearestPowerOfTwo + blockSize - 1) / blockSize);
            shiftRight<<<blocksForShift, blockSize>>>(n, inputBuf, outputBuf);

            timer().endGpuTimer();

            cudaMemcpy(odata, outputBuf, nearestPowerOfTwo * sizeof(int), cudaMemcpyDeviceToHost);
            cudaFree(outputBuf);
            cudaFree(inputBuf);
        }

        __global__ void naiveScan(int n, int depth, const int* inputBuf, int* outputBuf) {
            int threadId = threadIdx.x + (blockDim.x * blockIdx.x); 
            if (threadId >= n) return;

            int stride = 1 << (depth - 1);
            if (threadId >= n - stride) {
                outputBuf[n - threadId - 1] = inputBuf[n - threadId - 1];
            }
            else {
                outputBuf[threadId + stride] = inputBuf[threadId] + inputBuf[threadId + stride];
            }

        }

        __global__ void shiftRight(int n, const int* inputBuf, int* outputBuf) {
            int threadId = threadIdx.x + (blockDim.x * blockIdx.x); 
            if (threadId >= n) return;

            if (threadId == 0) {
                outputBuf[threadId] = 0;
                return;
            }

            outputBuf[threadId] = inputBuf[threadId - 1];
        }
    }
}

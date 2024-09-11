#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"
#include <algorithm>

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
            timer().startGpuTimer();

            int* outputBuf;
            int* inputBuf;
            cudaMalloc((void**)&outputBuf, n * sizeof(int));
            cudaMalloc((void**)&inputBuf, n * sizeof(int));
            cudaMemcpy(inputBuf, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            
            for (int depth = 1; depth <= ilog2ceil(n); ++depth) {
                int stride = static_cast<int>(pow(2, depth - 1));
                cudaMemcpy(outputBuf, inputBuf, sizeof(int) * stride, cudaMemcpyDeviceToDevice);

                int blockSize = std::min(n - stride, 1024); // cap at 1024 threads, hardware limitation.
                dim3 blocksPerGrid((n + blockSize - 1) / blockSize); // note integer division
                naiveScan<<<blocksPerGrid, blockSize>>>(n, depth, inputBuf, outputBuf);

                std::swap(outputBuf, inputBuf);
            }

            // Convert inclusive scan to exclusive scan
            int blockSize = std::min(n, 1024);
            dim3 blocksForShift((n + blockSize - 1) / blockSize);
            shiftRight<<<blocksForShift, blockSize>>>(n, inputBuf, outputBuf);

            cudaMemcpy(odata, outputBuf, n * sizeof(int), cudaMemcpyDeviceToHost);
            cudaFree(outputBuf);
            cudaFree(inputBuf);
            timer().endGpuTimer();
        }

        __global__ void naiveScan(int n, int depth, const int* inputBuf, int* outputBuf) {
            int threadId = threadIdx.x + (blockDim.x * blockIdx.x); 
            int stride = 1 << (depth - 1);
            if ((threadId) >= n - stride) return;


            outputBuf[threadId + stride] = inputBuf[threadId] + inputBuf[threadId + stride];
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

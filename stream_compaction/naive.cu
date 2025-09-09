#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"

#include <iostream>

namespace StreamCompaction {
    namespace Naive {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        //naive inclusive scan
        __global__ void naiveScanStep(int n, int d, const int* input, int* output) {
            int k = threadIdx.x + blockIdx.x * blockDim.x;
            if (k >= n) return;

            if (k >= (1 << (d - 1))) {
                output[k] = input[k - (1 << (d - 1))] + input[k];
            }
            else {
                output[k] = input[k];
            }
        }

        void scan(int n, int* odata, const int* idata) {

            int* dev_ping;
            int* dev_pong;

            cudaMalloc(&dev_ping, n * sizeof(int));
            cudaMalloc(&dev_pong, n * sizeof(int));

            cudaMemcpy(dev_ping, idata, n * sizeof(int), cudaMemcpyHostToDevice);

            int blockSize = 1024;
            int numBlocks = (n + blockSize - 1) / blockSize;

            int depth = ilog2ceil(n);

            timer().startGpuTimer();
            for (int d = 1; d <= depth; d++) {
                naiveScanStep << <numBlocks, blockSize >> > (n, d, dev_ping, dev_pong);

                // Swap buffers
                std::swap(dev_ping, dev_pong);
            }
            cudaDeviceSynchronize();
            timer().endGpuTimer();

            // dev_ping now has the inclusive scan result
            // Convert to exclusive scan
            cudaMemcpy(odata + 1, dev_ping, (n - 1) * sizeof(int), cudaMemcpyDeviceToHost);
            odata[0] = 0;

            cudaFree(dev_ping);
            cudaFree(dev_pong);

        }
    }
}

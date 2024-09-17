#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"

namespace StreamCompaction {
    namespace Naive {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }
        // TODO: __global__
        __global__ void naiveScanKernel(int* odata, const int* idata, int n, int stride) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < n) {
                if (idx >= stride) {
                    odata[idx] = idata[idx - stride] + idata[idx];
                }
                else {
                    odata[idx] = idata[idx];
                }
            }
        }

        /**
        * Performs prefix-sum (aka scan) on idata, storing the result into odata.
        */
        void scan(int n, int* odata, const int* idata) {
            timer().startGpuTimer();

            int* d_data1, * d_data2;
            cudaMalloc(&d_data1, n * sizeof(int));
            cudaMalloc(&d_data2, n * sizeof(int));

            cudaMemcpy(d_data1, idata, n * sizeof(int), cudaMemcpyHostToDevice);

            int* input = d_data1;
            int* output = d_data2;

            int numThreads = 256;
            int numBlocks = (n + numThreads - 1) / numThreads;

            for (int stride = 1; stride < n; stride *= 2) {
                naiveScanKernel << <numBlocks, numThreads >> > (output, input, n, stride);
                std::swap(input, output);
            }

            if (input != d_data1) {
                cudaMemcpy(d_data1, d_data2, n * sizeof(int), cudaMemcpyDeviceToDevice);
            }

            // Shift elements to the right to make it an exclusive scan
            cudaMemcpy(d_data2, d_data1, n * sizeof(int), cudaMemcpyDeviceToDevice);
            cudaMemset(d_data1, 0, sizeof(int)); // Set first element to 0
            cudaMemcpy(d_data1 + 1, d_data2, (n - 1) * sizeof(int), cudaMemcpyDeviceToDevice);

            cudaMemcpy(odata, d_data1, n * sizeof(int), cudaMemcpyDeviceToHost);

            cudaFree(d_data1);
            cudaFree(d_data2);

            timer().endGpuTimer();
        }
    }
}

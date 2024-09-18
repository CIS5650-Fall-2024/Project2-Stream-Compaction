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
        __global__ void kern_Naive_scan(int n, int offset, int* idata, int* odata)
        {
            int index = (blockIdx.x * blockDim.x) + threadIdx.x;

            if (index >= n) return;

            if (index >= offset) {
                odata[index] = idata[index] + idata[index - offset];
            }
            else {
                odata[index] = idata[index];
            }
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            int* dev_A, * dev_B;
            dim3 fullBlocksPerGrid((n + blockSize - 1) / blockSize);

            // Allocate device memory
            cudaMalloc((void**)&dev_A, n * sizeof(int));
            checkCUDAError("cudaMalloc dev_A failed!");
            cudaMalloc((void**)&dev_B, n * sizeof(int));
            checkCUDAError("cudaMalloc dev_B failed!");

            // Copy input data from host to device
            cudaMemcpy(dev_A, idata, n * sizeof(int), cudaMemcpyHostToDevice);

            timer().startGpuTimer();

            // Get the number of steps required
            int steps = ilog2ceil(n);

            for (int d = 1; d <= steps; d++) {
                int offset = 1 << (d - 1);  // Compute 2^(d-1)

                kern_Naive_scan << <fullBlocksPerGrid, blockSize >> > (n, offset, dev_A, dev_B);
                checkCUDAError("kern_Naive_scan failed!");

                cudaDeviceSynchronize();  // Wait for all threads to finish before the next step

                // ping-pong buffer to avoid race conditions
                std::swap(dev_A, dev_B);
            }

            timer().endGpuTimer();

            // Copy the result back to the host
            odata[0] = 0;
            cudaMemcpy(odata + 1, dev_A, n * sizeof(int), cudaMemcpyDeviceToHost);
            checkCUDAError("cudaMemcpy odata failed!");

            // Free device memory
            cudaFree(dev_A);
            cudaFree(dev_B);
        }
    }
}

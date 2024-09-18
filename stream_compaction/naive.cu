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

        // CUDA Kernel for naive scan
        __global__ void scanKernel(int n, int* odata, const int* idata, int offset) {
            int index = threadIdx.x + (blockIdx.x * blockDim.x);

            // validation check
            if (index >= n) {
                return;
            }

            // For each element, update the prefix sum using the offset
            if (index >= offset) {
                odata[index] = idata[index] + idata[index - offset];
            }
            else {
                odata[index] = idata[index]; // Elements before the offset remain the same
            }
        }



        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            // TODO

            // Allocate device memory
            int* dev_idata, * dev_odata;
            cudaMalloc((void**)&dev_idata, n * sizeof(int));
            cudaMalloc((void**)&dev_odata, n * sizeof(int));

            // Copy data from host to device
            cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);

            // Set up block and grid sizes
            int blockSize = 128; 
            int numBlocks = (n + blockSize - 1) / blockSize;
            dim3 fillBlockPerGrid(numBlocks);

            // Start GPU timer
            timer().startGpuTimer();

            // Perform naive scan
            for (int offset = 1; offset < n; offset *= 2) { //offset = 2^(d - 1)
                // Launch kernel
                scanKernel <<<fillBlockPerGrid, blockSize>>> (n, dev_odata, dev_idata, offset);

                ///This line should be added immediately after the kernel launch to 
                // ensure all threads have completed their work before copying the data back to dev_idat
                cudaDeviceSynchronize();

                // Copy odata back to idata for the next iteration
                cudaMemcpy(dev_idata, dev_odata, n * sizeof(int), cudaMemcpyDeviceToDevice);
            }

            // Stop GPU timer
            timer().endGpuTimer();

            // Copy results from device to host
            cudaMemcpy(odata, dev_odata, n * sizeof(int), cudaMemcpyDeviceToHost);

            // Free device memory
            cudaFree(dev_idata);
            cudaFree(dev_odata);

        }
    }
}

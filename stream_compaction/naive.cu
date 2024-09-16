#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"

#define blockSize 128 // Default is 128

namespace StreamCompaction {
    namespace Naive {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        __global__ void init_buff(int n, int *odata, const int *idata) {
            int i = threadIdx.x + (blockIdx.x * blockDim.x);

            if (i == 0) {
                odata[i] = 0;
                return;
            } 

            if (i < n) {
                odata[i] = idata[i - 1];
            } 
        }

        __global__ void dev_scan(int pow_d, int *odata, const int *idata) {
            int k = threadIdx.x + (blockIdx.x * blockDim.x);

            if (k >= pow_d) {
                odata[k] = idata[k - pow_d] + idata[k];
            }
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            timer().startGpuTimer();
            
            int *dev_buff1, *dev_buff2;

            // Allocate memory on the device
            cudaMalloc((void**)&dev_buff1, n * sizeof(int));  
            cudaMalloc((void**)&dev_buff2, n * sizeof(int));  

            // Copy data to the buffer for initialisation
            cudaMemcpy(dev_buff1, idata, n * sizeof(int), cudaMemcpyHostToDevice);

            // Set up the grid and block sizes
            dim3 fullBlocksPerGrid((n + blockSize - 1) / blockSize);

            // Because we want the first element to be 0 and the rest filled with the original data, we need to fill the values explicitly
            init_buff << <fullBlocksPerGrid, blockSize >> > (n, dev_buff2, dev_buff1);
            cudaMemcpy(dev_buff1, dev_buff2, n * sizeof(int), cudaMemcpyDeviceToDevice);

            for (int d = 1; d <= ilog2ceil(n); d++) {
                int pow_d = 1 << (d - 1);
                dev_scan << <fullBlocksPerGrid, blockSize >> > (pow_d, dev_buff2, dev_buff1);
                // cudaDeviceSynchronize();  // Ensure the kernel completes
                cudaMemcpy(dev_buff1, dev_buff2, n * sizeof(int), cudaMemcpyDeviceToDevice);
            }

            cudaMemcpy(odata, dev_buff1, n * sizeof(int), cudaMemcpyDeviceToHost);

            cudaFree(dev_buff1);
            cudaFree(dev_buff2);

            timer().endGpuTimer();
        }
    }
}

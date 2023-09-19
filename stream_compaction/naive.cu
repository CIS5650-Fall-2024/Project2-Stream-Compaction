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

        __global__ void naiveScanKernel(int n, int d, int* odata, const int* idata) {
          int k = threadIdx.x + (blockIdx.x * blockDim.x);
          if (k >= n) {
            return;
          }

          if (k >= (1 << (d - 1))) {
            odata[k] = idata[k - (1 << (d - 1))] + idata[k];
          }
          else {
            odata[k] = idata[k];
          }
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            timer().startGpuTimer();
            // TODO
            int* dev_idata;
            int* dev_odata;

            // Allocate device memory
            cudaMalloc((void**)&dev_idata, n * sizeof(int));
            cudaMalloc((void**)&dev_odata, n * sizeof(int));

            // Copy data to device
            cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);

            // Set up execution parameters
            int blockSize = 128;  
            int gridSize = (n + blockSize - 1) / blockSize;

            for (int d = 1; d <= ilog2ceil(n); ++d) {
              // Swap pointers
              if (d > 1) std::swap(dev_idata, dev_odata);

              // Launch kernel
              naiveScanKernel << <gridSize, blockSize >> > (n, d, dev_odata, dev_idata);

              // wait all computing finished
              cudaDeviceSynchronize();
            }

            // Copy results back to host
            cudaMemcpy(odata + 1, dev_odata , (n - 1) * sizeof(int), cudaMemcpyDeviceToHost);


            // manually convert it into exclusive scan
            odata[0] = 0;

            cudaFree(dev_idata);
            cudaFree(dev_odata);
            
            timer().endGpuTimer();
        }
    }
}

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
        __global__ void kernNaiveScan(int n, int offset, int *out, const int *in) {
          int k = index1D; 

          if (k < 1 || k >= n) {
            return; 
          }

          if (k >= offset) {
            out[k] = in[k - offset] + in[k];
          }
          else {
            out[k] = in[k]; 
          }
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            int arrSize = n; 
            if (!((n & (n - 1)) == 0)) {  // if n is not a power of 2, pad the array to next power of 2
              arrSize = 1 << ilog2ceil(n); 
            }

            int* dev_A = nullptr; 
            int* dev_B = nullptr; 

            // allocate some arrays on device
            cudaMalloc((void**)&dev_A, arrSize * sizeof(int));
            checkCUDAError("cudaMalloc dev_A failed!");

            cudaMalloc((void**)&dev_B, arrSize * sizeof(int));
            checkCUDAError("cudaMalloc dev_B failed!");

            cudaMemset(dev_A, 0, arrSize * sizeof(int)); 
            checkCUDAError("cudaMemset dev_A failed!");

            cudaMemset(dev_B, 0, arrSize * sizeof(int));
            checkCUDAError("cudaMemset dev_B failed!");

            // copy to A, which initially has input data
            // also offset by one to do an exclusive scan
            cudaMemcpy(dev_A + 1, idata, (n - 1) * sizeof(int), cudaMemcpyHostToDevice);
            checkCUDAError("cudaMemcpy dev_A failed!");

            timer().startGpuTimer();
            // run kernel over depth
            for (int d = 1; d <= arrSize; d <<= 1) { 
              kernNaiveScan <<<blocksPerGrid(arrSize), BLOCKSIZE>>>(arrSize, d, dev_B, dev_A);
              checkCUDAError("kernNaiveScan failed!");

              // swap the buffers
              int* swap = dev_A; 
              dev_A = dev_B; 
              dev_B = swap; 
            }
            timer().endGpuTimer();

            odata[0] = 0; // first element is always the identity

            // copy from device to host. shifting one to the right to do an exclusive scan
            cudaMemcpy(odata, dev_A, n * sizeof(int), cudaMemcpyDeviceToHost);
            checkCUDAError("cudaMemcpy dev_A to odata failed!");

            // free device memory
            cudaFree(dev_A);
            cudaFree(dev_B); 
            
        }
    }
}

#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"
#include <iostream>

namespace StreamCompaction {
    namespace Efficient {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        __global__ void kernEfficientScan(int n, int* data) {
          int idx = threadIdx.x;
          if (idx >= (n >> 1)) {
            return; 
          }
          extern __shared__ int temp[]; 
          temp[2 * idx]     = data[2 * idx];
          temp[2 * idx + 1] = data[2 * idx + 1];
          
          // up sweep 
          for (int depth = 1; depth < n; depth <<= 1) {
            __syncthreads();
            int offset = idx * (depth << 1);  // k * 2^(d+1)
            if (offset < n) {
              temp[offset + (depth << 1) - 1] += temp[offset + depth - 1];
            }
          }

          temp[n - 1] = 0;
          for (int depth = (n >> 1); depth >= 1; depth >>= 1) {
            __syncthreads(); 
            int offset = idx * (depth << 1); 
            if (offset < n) {
              int t = temp[offset + depth - 1];
              temp[offset + depth - 1] = temp[offset + (depth << 1) - 1];
              temp[offset + (depth << 1) - 1] += t; 
            }
          }
          __syncthreads(); 
          data[2 * idx] = temp[2 * idx];
          data[2 * idx + 1] = temp[2 * idx + 1];
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            timer().startGpuTimer();

            int* dev_data = nullptr; 

            cudaMalloc((void**)&dev_data, sizeof(int) * n); 
            cudaMemcpy(dev_data, idata, sizeof(int) * n, cudaMemcpyHostToDevice); 

            int numBlocks = blocksPerGrid((n >> 1));
            int numThreads = BLOCKSIZE; 

            // call kernel 
            kernEfficientScan<<<numBlocks, numThreads, n>>>(n, dev_data);

            cudaMemcpy(odata, dev_data, sizeof(int) * n, cudaMemcpyDeviceToHost); 
            cudaFree(dev_data); 

            cudaDeviceSynchronize(); 
            timer().endGpuTimer();
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
            
            // todo
            timer().endGpuTimer();
            return -1;
        }
    }
}

#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"
#include <iostream>

#define globalIdx ((blockIdx.x * blockDim.x) + threadIdx.x)

namespace StreamCompaction {
    namespace Efficient {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        __global__ void kernEfficientScanMultiBlock(int n, int* data, int* sum) {
          extern __shared__ int temp[];
          temp[2 * threadIdx.x] = data[2 * globalIdx];
          temp[2 * threadIdx.x + 1] = data[2 * globalIdx + 1];

          // up sweep 
          for (int depth = 1; depth < n; depth <<= 1) {
            __syncthreads();
            int offset = threadIdx.x * (depth << 1);  // k * 2^(d+1)
            if (offset < n) {
              temp[offset + (depth << 1) - 1] += temp[offset + depth - 1];
            }
          }

          temp[n - 1] = 0;
          for (int depth = (n >> 1); depth >= 1; depth >>= 1) {
            __syncthreads();
            int offset = threadIdx.x * (depth << 1);
            if (offset < n) {
              int t = temp[offset + depth - 1];
              temp[offset + depth - 1] = temp[offset + (depth << 1) - 1];
              temp[offset + (depth << 1) - 1] += t;
            }
          }
          __syncthreads();

          // ensure we get an inclusive scan as our result
          if ((2 * threadIdx.x + 1) == n - 1) {
            data[2 * globalIdx] = temp[2 * threadIdx.x + 1];
            data[2 * globalIdx + 1] += temp[2 * threadIdx.x + 1];
            sum[blockIdx.x] = data[2 * globalIdx + 1];
            return; 
          }

          data[2 * globalIdx] = temp[2 * threadIdx.x + 1];
          data[2 * globalIdx + 1] = temp[2 * threadIdx.x + 2];
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

        __global__ void kernBlockIncrements(int n, int* data, int* sum) {
          int idx = 2 * ((blockIdx.x * blockDim.x) + threadIdx.x); 
          if (idx >= n) {
            return; 
          }
          data[idx] += sum[blockIdx.x];
          data[idx + 1] += sum[blockIdx.x];
        }

        void _scan(int n, int* dev_data) {
          int numBlocks = blocksPerGrid((n >> 1));  // enough blocks that can handle 2 elements per thread, up to n elements
          int numThreads = BLOCKSIZE;

          if (numBlocks == 1) {
            kernEfficientScan<<<numBlocks, numThreads, n * sizeof(int)>>>(n, dev_data);
            checkCUDAError("kernEfficientScan failed");
          }
          else {
            int* dev_sum = nullptr; 
            cudaMalloc((void**)&dev_sum, numBlocks * sizeof(int)); 
            checkCUDAError("cudaMalloc dev_sum failed"); 

            int numElementsPerBlock = numThreads << 1; 

            kernEfficientScanMultiBlock<<<numBlocks, numThreads, numElementsPerBlock * sizeof(int)>>>(numElementsPerBlock, dev_data, dev_sum); 
            checkCUDAError("kernEfficientScanMultiBlock failed");

            // perform (exclusive) scan
            _scan(numBlocks, dev_sum); 

            // perform sums on dev_data
            kernBlockIncrements<<<numBlocks, numThreads>>>(n, dev_data, dev_sum);
            checkCUDAError("kernBlockIncrements dev_sum failed");

            // inclusive to exclusive scan by shifting the results and inserting identity
            int* dev_temp = nullptr; 
            cudaMalloc((void**)&dev_temp, n * sizeof(int)); 
            checkCUDAError("cudaMalloc dev_temp failed");

            cudaMemset(dev_temp, 0, 1 * sizeof(int)); 
            checkCUDAError("cudaMemset dev_temp failed");

            cudaMemcpy(dev_temp + 1, dev_data, (n - 1) * sizeof(int), cudaMemcpyDeviceToDevice);
            checkCUDAError("cudaMemcpy dev_data to dev_temp faile");

            cudaMemcpy(dev_data, dev_temp, n * sizeof(int), cudaMemcpyDeviceToDevice); 
            checkCUDAError("cudaMemcpy dev_temp to dev_data failed");

            // ??? profit
            cudaFree(dev_sum); 
            cudaFree(dev_temp); 
          }
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            timer().startGpuTimer();

            int arrSize = n;
            if (n & (n - 1)) {  // if n is not a power of 2, pad the array to next power of 2
              arrSize = 1 << ilog2ceil(n);
            }

            int* dev_data = nullptr; 

            cudaMalloc((void**)&dev_data, sizeof(int) * arrSize);
            checkCUDAError("cudaMalloc dev_data failed");

            cudaMemset(dev_data, 0, sizeof(int) * arrSize); 
            checkCUDAError("cudaMalloc dev_data failed");

            cudaMemcpy(dev_data, idata, sizeof(int) * n, cudaMemcpyHostToDevice);
            checkCUDAError("cudaMemcpy dev_data failed");

            _scan(arrSize, dev_data); 

            cudaMemcpy(odata, dev_data, sizeof(int) * n, cudaMemcpyDeviceToHost);
            checkCUDAError("cudaMemcpy dev_data failed");

            cudaFree(dev_data);
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

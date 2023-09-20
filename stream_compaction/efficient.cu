#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"
#include "naive.h"

namespace StreamCompaction {
    namespace Efficient {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        __global__ void upSweep(int n, int d, int* data) {
          int k = threadIdx.x + blockIdx.x * blockDim.x;
          if (k >= n) return;
          int offset = 1 << (d + 1);

          if (k % offset == 0) {
            int left = (k + offset - 1);
            int right = k + (offset >> 1) - 1;
            if (left < n) {
              data[left] += data[right];
            }
          }
        }

        __global__ void downSweep(int n, int d, int* data) {
          int k = threadIdx.x + blockIdx.x * blockDim.x;
          int offset = 1 << (d + 1);

          if (k % offset == 0) {
            int b = k + (offset >> 1) - 1;
            int a = k + offset - 1;

            if (a < n) {
              int temp = data[b];
              data[b] = data[a];
              data[a] += temp;
            }
          }
        }

        __global__ void upSweepV2(int n, int stride, int* data) {
          int k = threadIdx.x + blockIdx.x * blockDim.x;
          if (k >= n) return;

          int left = stride * (k + 1) - 1;
          int right = stride * k + (stride >> 1) - 1;
          if (left < n) {
            data[left] += data[right];
          }
        }

        __global__ void downSweepV2(int n, int stride, int* data) {
          int k = threadIdx.x + blockIdx.x * blockDim.x;
          if (k >= n) return;

          // Compute the indices for the left and right elements to be operated on
          int a = stride * (k + 1) - 1;
          int b = stride * k + (stride >> 1) - 1;

          if (a < n) {
            int temp = data[b];
            data[b] = data[a];
            data[a] += temp;
          }
        }

        __global__ void blockScan(int* data, int* blockSums, int n, int D_max) {
          extern __shared__ int sdata[];
          int tid = threadIdx.x;
          int gid = blockIdx.x * blockDim.x + threadIdx.x;

          if (gid >= n) return;
          // Load input into shared memory.
          sdata[tid] = (gid < n) ? data[gid] : 0;
          __syncthreads();

          // Up-sweep (Reduce)
          int stride;
          for (int d = 0; d <= D_max; ++d) {
            stride = 1 << (d + 1);

            if (tid < ((blockDim.x + stride - 1) / stride)) {
              int right = stride * tid + (stride >> 1) - 1;
              int left = stride * (tid + 1) - 1;
              if (left < blockDim.x) sdata[left] += sdata[right];
            }
            __syncthreads();
          }

           //Clear last element for downsweep
          int lastSum;
          if (tid == blockDim.x - 1) {
            lastSum = sdata[blockDim.x - 1];
            sdata[blockDim.x - 1] = 0;
            if (blockSums != nullptr) {
              blockSums[blockIdx.x] = lastSum;
            }
          }
          __syncthreads();

          //// Down-sweep
          for (int d = D_max; d >= 0; --d) {
            stride = 1 << (d + 1);

            if (tid < ((blockDim.x + stride - 1) / stride)) {
              // Compute the indices for the left and right elements to be operated on
              int a = stride * (tid + 1) - 1;
              int b = stride * tid + (stride >> 1) - 1;

              if (a < blockDim.x) {
                int temp = sdata[b];
                sdata[b] = sdata[a];
                sdata[a] += temp;
              }
            }
            __syncthreads();
          }

          // Write results to output array.
          //convert the exclusive scan into inclusive one
          if (tid < blockDim.x - 1) {
            data[gid] = sdata[tid + 1];
          }
          else if (tid == blockDim.x - 1) {
            data[gid] = lastSum;
          }

          //data[gid] = sdata[tid];
        }

        __global__ void addBlockSums(int* data, int* blockSums, int n) {
          int gid = blockIdx.x * blockDim.x + threadIdx.x;
          if (gid >= n) return;
          
          int blockSum;
          if (blockIdx.x >= 1) {
            blockSum = blockSums[blockIdx.x - 1];
            data[gid] += blockSum;
          }
        }

        void scanShared(int n, int* odata, const int* idata) {
          int d = ilog2ceil(n) - 1;
          int extendLength = 1 << (d + 1);

          int blockSize = 128; 
          int numBlocks = (extendLength + blockSize - 1) / blockSize;

          int d_max = ilog2ceil(blockSize) - 1;
          int* dev_data;
          int* dev_blockSums;

          cudaMalloc((void**)&dev_data, extendLength * sizeof(int));
          cudaMemset(dev_data, 0, extendLength * sizeof(int));
          cudaMemcpy(dev_data, idata, n * sizeof(int), cudaMemcpyHostToDevice);

          cudaMalloc((void**)&dev_blockSums, numBlocks * sizeof(int));
          cudaMemset(dev_blockSums, 0, numBlocks * sizeof(int));

          int* dev_o_blockSums;
          cudaMalloc((void**)&dev_o_blockSums, numBlocks * sizeof(int));
          cudaMemset(dev_o_blockSums, 0, numBlocks * sizeof(int));

          nvtxRangePushA("scanShared");
          timer().startGpuTimer();
          //Step 1: Perform scan on individual blocks and record the block sums.
          blockScan << <numBlocks, blockSize >> > (dev_data, dev_blockSums, extendLength, d_max);
          checkCUDAError("blockScan");

          //Step 1.1: Perform scan on block sums.
          int newblockSize = 512;
          int gridSize = (numBlocks + newblockSize - 1) / newblockSize;
          
          for (int d = 1; d <= ilog2ceil(numBlocks); ++d) {
            // Swap pointers
            if (d > 1) std::swap(dev_blockSums, dev_o_blockSums);

            // Launch kernel
            StreamCompaction::Naive::naiveScanKernel << <gridSize, newblockSize >> > (numBlocks, d, dev_o_blockSums, dev_blockSums);
            checkCUDAError("naiveScanKernel");
            // wait all computing finished
            //cudaDeviceSynchronize();
          }

          //// Step 2: Add the block sums to the scanned array.
          addBlockSums << <numBlocks, blockSize >> > (dev_data, dev_o_blockSums, n);
          checkCUDAError("addBlockSums");
          timer().endGpuTimer();
          nvtxRangePop();

          // Copy results back to host
          cudaMemcpy(odata + 1, dev_data, (n - 1) * sizeof(int), cudaMemcpyDeviceToHost);

          // manually convert it into exclusive scan
          odata[0] = 0;

          cudaFree(dev_data);
          cudaFree(dev_blockSums);
          cudaFree(dev_o_blockSums);
        }


        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            int d = ilog2ceil(n) - 1;

            int extendLength = 1 << (d + 1);
            // Allocate device memory
            int* dev_data;
            cudaMalloc((void**)&dev_data, extendLength * sizeof(int));

            // set the cuda memory
            cudaMemset(dev_data, 0, extendLength * sizeof(int));
            cudaMemcpy(dev_data, idata, n * sizeof(int), cudaMemcpyHostToDevice);

            // Set up execution parameters
            int blockSize = 256;
            int initialBlockSize = blockSize;

            nvtxRangePushA("workefficient");
            timer().startGpuTimer();
            // ------------------------ Version 2.0 -------------------------------------------
            // 
            // 
            // index optimizing, thus wraps divergance optimizing
            // Up-sweep
            int stride = 0;
            int threadsNeeded = 0;
            
            int gridSize;
            for (int i = 0; i <= d; ++i) {
              stride = 1 << (1 + i);
              threadsNeeded = (extendLength + stride - 1) / stride;
              blockSize = std::max(1, initialBlockSize / stride);
              gridSize = (threadsNeeded + blockSize - 1) / blockSize;
              upSweepV2 << <gridSize, blockSize >> > (extendLength, stride, dev_data);
              checkCUDAError("upSweepV2");
            }

            // Clear the last element
            cudaMemset(&dev_data[extendLength - 1], 0, sizeof(int));

            //// Down-sweep
            for (int i = d; i >= 0; --i) {
              stride = 1 << (1 + i);
              threadsNeeded = (extendLength + stride - 1) / stride;
              blockSize = std::max(1, initialBlockSize / stride);
              gridSize = (threadsNeeded + blockSize - 1) / blockSize;
              downSweepV2 << <gridSize, blockSize >> > (extendLength, stride, dev_data);
              checkCUDAError("downSweepV2");
            }
            //_______________________Version 2.0 ___________________________________________________
            timer().endGpuTimer();
            nvtxRangePop();

            cudaDeviceSynchronize();
            // Copy results back to host
            cudaMemcpy(odata, dev_data, n * sizeof(int), cudaMemcpyDeviceToHost);
            checkCUDAError("cudaMemcpy");

            // Free device memory
            cudaFree(dev_data);
            checkCUDAError("cudaFree");
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
            int d = ilog2ceil(n) - 1;

            int extendLength = 1 << (d + 1);
            // Allocate device memory
            int* dev_idata;
            int* dev_odata;
            int* dev_bools;
            int* dev_indices;
            cudaMalloc((void**)&dev_idata, extendLength * sizeof(int));
            checkCUDAError("cudaMalloc");
            cudaMalloc((void**)&dev_odata, n * sizeof(int));
            checkCUDAError("cudaMalloc");
            cudaMalloc((void**)&dev_bools, extendLength * sizeof(int));
            checkCUDAError("cudaMalloc");
            cudaMalloc((void**)&dev_indices, extendLength * sizeof(int));
            checkCUDAError("cudaMalloc");
            
            // Set up execution parameters
            int blockSize = 128;
            int initialBlockSize = blockSize;
            int gridSize = (extendLength + blockSize - 1) / blockSize;

            cudaMemset(dev_bools, 0, extendLength * sizeof(int));
            checkCUDAError("cudaMemset");
            cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            cudaMemset(dev_idata + n, 0, (extendLength - n) * sizeof(int));
            checkCUDAError("cudaMemset");

            timer().startGpuTimer();

            StreamCompaction::Common::kernMapToBoolean << <gridSize, blockSize >> > (extendLength, dev_bools, dev_idata);
            checkCUDAError("kernMapToBoolean");

            cudaMemcpy(dev_indices, dev_bools, extendLength * sizeof(int), cudaMemcpyDeviceToDevice);
            // Up-sweep
            int stride = 0;
            //for (int i = 0; i <= d; ++i) {
            //  stride = 1 << (1 + i);
            //  upSweep << <gridSize, blockSize >> > (extendLength, i, dev_indices);
            //}
            int threadsNeeded;
            for (int i = 0; i <= d; ++i) {
              stride = 1 << (1 + i);
              threadsNeeded = (extendLength + stride - 1) / stride;
              blockSize = std::max(1, initialBlockSize / stride);
              gridSize = (threadsNeeded + blockSize - 1) / blockSize;
              upSweepV2 << <gridSize, blockSize >> > (extendLength, stride, dev_indices);
              checkCUDAError("upSweepV2");
            }

            // Clear the last element
            cudaMemset(&dev_indices[extendLength - 1], 0, sizeof(int));

            // Down-sweep
            //for (int i = d; i >= 0; --i) {
            //  stride = 1 << (1 + i);
            //  downSweep << <gridSize, blockSize >> > (extendLength, i, dev_indices);
            //}

            //// Down-sweep
            for (int i = d; i >= 0; --i) {
              stride = 1 << (1 + i);
              threadsNeeded = (extendLength + stride - 1) / stride;
              blockSize = std::max(1, initialBlockSize / stride);
              gridSize = (threadsNeeded + blockSize - 1) / blockSize;
              downSweepV2 << <gridSize, blockSize >> > (extendLength, stride, dev_indices);
              checkCUDAError("downSweepV2");
            }

            gridSize = (extendLength + initialBlockSize - 1) / initialBlockSize;
            StreamCompaction::Common::kernScatter << <gridSize, initialBlockSize >> > (extendLength, dev_odata, dev_idata, dev_bools, dev_indices);
            //StreamCompaction::Common::kernScatter << <gridSize, blockSize >> > (extendLength, dev_odata, dev_idata, dev_bools, dev_indices);
            checkCUDAError("kernScatter");
            

            // Copy results back to host
            cudaMemcpy(odata, dev_odata, n * sizeof(int), cudaMemcpyDeviceToHost);
            checkCUDAError("cudaMemcpy");

            cudaFree(dev_idata);
            cudaFree(dev_odata);
            cudaFree(dev_bools);
            cudaFree(dev_indices);
            checkCUDAError("cudaFree");
            timer().endGpuTimer();

            int i;
            for (i = 0; i < n; i++) {
              if (odata[i] == 0) return i;
            }

            if (i == n) return n;
            return -1;
        }
    }
}

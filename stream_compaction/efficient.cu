#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

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

        __global__ void upSweepV2(int n, int d, int* data) {
          int k = threadIdx.x + blockIdx.x * blockDim.x;
          if (k >= n) return;
          int stride = 1 << (d + 1);

          if (k < (n / stride)) {
            int left = stride * (k + 1) - 1;
            int right = stride * k + (stride >> 1) - 1;
            data[left] += data[right];
          }
        }

        __global__ void downSweepV2(int n, int d, int* data) {
          int k = threadIdx.x + blockIdx.x * blockDim.x;
          int stride = 1 << (d + 1);

          if (k < (n / stride)) {
            // Compute the indices for the left and right elements to be operated on
            int a = stride * (k + 1) - 1;
            int b = stride * k + (stride >> 1) - 1;

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

            if (tid < (blockDim.x / stride)) {
              
              int right = stride * tid + (stride >> 1) - 1;
              int left = stride * (tid + 1) - 1;
              sdata[left] += sdata[right];
            }
            __syncthreads();
          }

          // Clear last element for downsweep
          int lastSum = sdata[blockDim.x - 1];
          if (tid == 0) {
            sdata[blockDim.x - 1] = 0;
          }
          __syncthreads();

          //// Down-sweep
          for (int d = D_max; d >= 0; --d) {
            stride = 1 << (d + 1);

            if (tid < (blockDim.x / stride)) {
              // Compute the indices for the left and right elements to be operated on
              int a = stride * (tid + 1) - 1;
              int b = stride * tid + (stride >> 1) - 1;

              int temp = sdata[b];
              sdata[b] = sdata[a];
              sdata[a] += temp;
            }
          }

          // Write results to output array.
          //convert the exclusive scan into inclusive one
          if (tid < blockDim.x - 1) {
            data[gid] = sdata[tid + 1];
          }
          else if (tid == blockDim.x - 1) {
            data[gid] = lastSum;
          }

          // Record the block sum for this block.
          if (tid == 0) {
            if (blockSums != nullptr) {
              blockSums[blockIdx.x] = lastSum;
            }
          }
        }

        __global__ void addBlockSums(int* data, int* blockSums, int n) {
          int gid = blockIdx.x * blockDim.x + threadIdx.x;
          if (gid >= n) return;
          
          int blockSum = 0;
          if (blockIdx.x >= 1) {
            for (int i = 0; i < blockIdx.x; i++) {
              blockSum += blockSums[i];
            }
            //blockSum = blockSums[blockIdx.x - 1];
            //data[gid] += blockSum;
          }
        }

        void scanShared(int n, int* datao, const int* datai) {
          int blockSize = 64; 
          int numBlocks = (n + blockSize - 1) / blockSize;

          int d_max = ilog2ceil(blockSize) - 1;
          int d_max_2 = ilog2ceil(numBlocks) - 1;
          int* dev_data;
          int* dev_blockSums;
          cudaMalloc(&dev_data, n * sizeof(int));
          cudaMalloc(&dev_blockSums, numBlocks * sizeof(int));

          cudaMemcpy(dev_data, datai, n * sizeof(int), cudaMemcpyHostToDevice);
          cudaMemset(dev_blockSums, 0, numBlocks);

          timer().startGpuTimer();
          // Step 1: Perform scan on individual blocks and record the block sums.
          blockScan << <numBlocks, blockSize >> > (dev_data, dev_blockSums, n, d_max);
          cudaGetLastError();

          // Step 1.1: Perform scan on block sums.
          //blockScan << <1, numBlocks >> > (dev_blockSums, nullptr, numBlocks, d_max_2);
          //cudaGetLastError();

          // Step 2: Add the block sums to the scanned array.
          addBlockSums << <numBlocks, blockSize >> > (dev_data, dev_blockSums, n);
          timer().endGpuTimer();

          // Copy results back to host
          cudaMemcpy(datao + 1, dev_data, (n - 1) * sizeof(int), cudaMemcpyDeviceToHost);

          // manually convert it into exclusive scan
          datao[0] = 0;

          cudaFree(dev_data);
          cudaFree(dev_blockSums);
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
            int blockSize = 128;
            int gridSize = (extendLength + blockSize - 1) / blockSize;

            timer().startGpuTimer();
            // ------------------------ Version 1.0 -------------------------------------------
            // 
            // 
            // This is my first trival and it is really slow because of wrap divergence and excess global memory access 
            //// Up-sweep
            //for (int i = 0; i <= d; ++i) {
            //  upSweep << <gridSize, blockSize >> > (extendLength, i, dev_data);
            //}

            //// Clear the last element
            //cudaMemset(&dev_data[extendLength - 1], 0, sizeof(int));

            //// Down-sweep
            //for (int i = d; i >= 0; --i) {
            //  downSweep << <gridSize, blockSize >> > (extendLength, i, dev_data);
            //}
            //_______________________Version 1.0 ___________________________________________________


            // ------------------------ Version 2.0 -------------------------------------------
            // 
            // 
            // index optimizing, thus wraps divergance optimizing
            // Up-sweep
            for (int i = 0; i <= d; ++i) {
              upSweepV2 << <gridSize, blockSize >> > (extendLength, i, dev_data);
            }

            // Clear the last element
            cudaMemset(&dev_data[extendLength - 1], 0, sizeof(int));

            // Down-sweep
            for (int i = d; i >= 0; --i) {
              downSweepV2 << <gridSize, blockSize >> > (extendLength, i, dev_data);
            }
            //_______________________Version 2.0 ___________________________________________________
            timer().endGpuTimer();


            // Copy results back to host
            cudaMemcpy(odata, dev_data, n * sizeof(int), cudaMemcpyDeviceToHost);

            // Free device memory
            cudaFree(dev_data);
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
            
            // TODO

            int d = ilog2ceil(n) - 1;

            int extendLength = 1 << (d + 1);
            // Allocate device memory
            int* dev_idata;
            int* dev_odata;
            int* dev_bools;
            int* dev_indices;
            cudaMalloc((void**)&dev_idata, extendLength * sizeof(int));
            cudaMalloc((void**)&dev_odata, extendLength * sizeof(int));
            cudaMalloc((void**)&dev_bools, extendLength * sizeof(int));
            cudaMalloc((void**)&dev_indices, extendLength * sizeof(int));
            
            // Set up execution parameters
            int blockSize = 128;
            int gridSize = (extendLength + blockSize - 1) / blockSize;

            cudaMemset(dev_bools, 0, extendLength);
            cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            cudaMemset(dev_idata + n, 0, (extendLength - n) * sizeof(int));

            timer().startGpuTimer();

            StreamCompaction::Common::kernMapToBoolean << <gridSize, blockSize >> > (n, dev_bools, dev_idata);

            cudaMemcpy(dev_indices, dev_bools, extendLength * sizeof(int), cudaMemcpyDeviceToDevice);
            // Up-sweep
            for (int i = 0; i <= d; ++i) {
              upSweepV2 << <gridSize, blockSize >> > (extendLength, i, dev_indices);
            }

            // Clear the last element
            cudaMemset(&dev_indices[extendLength - 1], 0, sizeof(int));

            // Down-sweep
            for (int i = d; i >= 0; --i) {
              downSweepV2 << <gridSize, blockSize >> > (extendLength, i, dev_indices);
            }

            StreamCompaction::Common::kernScatter << <gridSize, blockSize >> > (n, dev_odata, dev_idata, dev_bools, dev_indices);

            timer().endGpuTimer();

            // Copy results back to host
            cudaMemcpy(odata, dev_odata, n * sizeof(int), cudaMemcpyDeviceToHost);

            cudaFree(dev_idata);
            cudaFree(dev_odata);
            cudaFree(dev_bools);
            cudaFree(dev_indices);

            

            int i;
            for (i = 0; i < n; i++) {
              if (odata[i] == 0) return i;
            }

            if (i == n) return n;
            return -1;
        }
    }
}

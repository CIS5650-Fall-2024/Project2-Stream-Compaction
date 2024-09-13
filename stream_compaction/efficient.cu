#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"
#include <algorithm>

namespace StreamCompaction {
    namespace Efficient {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        const int MAX_BLOCK_SIZE = 1024;

        void scan(int n_padded, int* dev_data) {
            int maxBlockSize = std::min((n_padded / 2), MAX_BLOCK_SIZE);
            int stride = 2 * maxBlockSize; // since each thread works on two entries of dev_data, 2x block size gives the stride to get between blocks of data in dev_data.
            dim3 blocksPerGrid = ((n_padded / 2) + maxBlockSize - 1) / maxBlockSize;

            int* stored_sums; // temp array used to last entry per block during upsweep. See kernZeroEntries and kernIncrement for use info.
            cudaMalloc((void**)&stored_sums, blocksPerGrid.x * sizeof(int));

            int blockSize_i = maxBlockSize;
            for (int depth = 0; depth < ilog2ceil(2 * maxBlockSize); ++depth) {
                kernUpSweep<<<blocksPerGrid, blockSize_i>>>(n_padded, stride, depth, dev_data);

                blockSize_i /= 2; // need fewer threads each iteration
                cudaDeviceSynchronize();
            }

            // Between upsweep and downsweep, zero out the last entry of every block (first storing off those entries for later use)
            int zeroEntriesBlockSize = 128;
            dim3 zeroEntriesBlocksPerGrid = (blocksPerGrid.x + zeroEntriesBlockSize - 1) / zeroEntriesBlockSize;
            kernZeroEntries<<<zeroEntriesBlocksPerGrid, zeroEntriesBlockSize>>> (blocksPerGrid.x, stride, dev_data, stored_sums);
            cudaDeviceSynchronize();

            // Keep blocks per grid constant, so we can handle arbitrarily sized arrays.
            // But grow blockSize on each iteration since we need more threads on each depth layer.
            blockSize_i = 1;
            for (int depth = ilog2ceil(2 * maxBlockSize) - 1; depth >= 0; --depth) {
                kernDownSweep<<<blocksPerGrid, blockSize_i>>>(n_padded, stride, depth, dev_data);

                blockSize_i *= 2; // need more threads each iteration
                cudaDeviceSynchronize();
            }

            // If the array didn't fit within a single block, we need to collect the individual block scan results, 
            // put them in an array, and scan that array. Then add the twice-scanned array as increments back to the original results.
            //
            // This needs to be done recursively to handle arbitrarily large arrays.
            if (n_padded > 2 * MAX_BLOCK_SIZE) {
                // (Recursively) scan the summed blocks array
                // Can use sum_data as both the input and output pointers for the scan. No issue writing over it.
                scan(blocksPerGrid.x, stored_sums);

                // Finally, add scanned sum values back to the original dev_data
                // In original scan, each thread handled 2 elements. In this step, each handles one, so we need 2x the blocks.
                dim3 kernBlocksPerGrid = 2 * blocksPerGrid.x;
                kernIncrement<<<kernBlocksPerGrid, maxBlockSize>>> (n_padded, dev_data, stored_sums);
                cudaDeviceSynchronize();
            }

            cudaFree(stored_sums);
        }

        /**
         * Wrapper around scan (to facilitate gpu timing and allocating things)
         */
        void scan(int n, int *odata, const int *idata) {
            int n_padded = pow(2, ilog2ceil(n)); // pad array to nearest power of two

            int* dev_data;
            cudaMalloc((void**)&dev_data, n_padded * sizeof(int));
            cudaMemcpy(dev_data, idata, n * sizeof(int), cudaMemcpyHostToDevice);

            // Pad input data to be a power of two, if needed.
            if (n < n_padded) {
                cudaMemset(dev_data + n, 0, (n_padded - n) * sizeof(int));
            }

            timer().startGpuTimer();
            scan(n_padded, dev_data);
            timer().endGpuTimer();

            cudaMemcpy(odata, dev_data, n_padded * sizeof(int), cudaMemcpyDeviceToHost);
            cudaFree(dev_data);
        }

        __global__ void kernUpSweep(int n, int stride, int depth, int* dev_data) {
            int threadId = threadIdx.x; 

            int twoToDepthPlusOne = (1 << (depth + 1));
            int twoToDepth = (1 << depth);
            // Since each block is a self contained scan, we calculate these indices w.r.t the local block thread index.
            // But then we offset by (blockDim.x * blockIdx.x) because the dev_data is for ALL blocks, so we need to access the right part.
            int leftChildIdx = (threadId * twoToDepthPlusOne) + twoToDepth - 1 + (stride * blockIdx.x);
            int rightChildIdx = (threadId * twoToDepthPlusOne) + twoToDepthPlusOne - 1 + (stride * blockIdx.x);

            if (rightChildIdx >= n) return;

            dev_data[rightChildIdx] += dev_data[leftChildIdx];
        }

        __global__ void kernZeroEntries(int n, int stride, int* dev_data, int* stored_sums) {
            int threadId = threadIdx.x + (blockDim.x * blockIdx.x);
            if (threadId >= n) return;

            int dev_data_idx = (threadId + 1) * stride - 1;
            stored_sums[threadId] = dev_data[dev_data_idx];
            dev_data[dev_data_idx] = 0;
        }

        __global__ void kernDownSweep(int n, int stride, int depth, int* dev_data) {
            int threadId = threadIdx.x; 

            int twoToDepthPlusOne = (1 << (depth + 1));
            int twoToDepth = (1 << depth);
            int blockLeftChildIdx = (threadId * twoToDepthPlusOne) + twoToDepth - 1;
            int globalLeftChildIdx = blockLeftChildIdx + (stride * blockIdx.x);
            int blockRightChildIdx = (threadId * twoToDepthPlusOne) + twoToDepthPlusOne - 1;
            int globalRightChildIdx = blockRightChildIdx + (stride * blockIdx.x);

            if (globalRightChildIdx >= n) return;

            int leftVal = dev_data[globalLeftChildIdx];
            dev_data[globalLeftChildIdx] = dev_data[globalRightChildIdx];
            dev_data[globalRightChildIdx] += leftVal;
        }

        /**
         * Kernel to add the scanned block sums back to the original array.
         * 
         * n here is the number of elements in the original input arary.
         */
        __global__ void kernIncrement(int n, int* dev_data, int* sum_data) {
            int threadId = threadIdx.x + (blockIdx.x * blockDim.x);
            if (threadId >= n) return;

            // The extra factor of 2 comes from the fact that we're using twice as many block here as in the original scan.
            int sum_data_idx = gridDim.x * threadId / (2 * n); // note: integer division here

            dev_data[threadId] += sum_data[sum_data_idx];
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
            // TODO
            timer().endGpuTimer();
            return -1;
        }
    }
}

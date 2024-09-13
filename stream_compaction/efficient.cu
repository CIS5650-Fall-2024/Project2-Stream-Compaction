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

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            int n_padded = pow(2, ilog2ceil(n)); // pad array to nearest power of two

            int* dev_data;
            cudaMalloc((void**) &dev_data, n_padded * sizeof(int));
            cudaMemcpy(dev_data, idata, n * sizeof(int), cudaMemcpyHostToDevice);

            // Pad input data to be a power of two, if needed.
            if (n < n_padded) {
                cudaMemset(dev_data + n, 0, (n_padded - n) * sizeof(int));
            }

            int blockSize = std::min((n_padded / 2), MAX_BLOCK_SIZE);
            dim3 blocksPerGrid = ((n_padded / 2) + blockSize - 1) / blockSize;

            int* stored_sums; // temp array used to last entry per block during upsweep. See kernBlockSums for more info.
            cudaMalloc((void**) &stored_sums, blocksPerGrid.x * sizeof(int));

            // timer().startGpuTimer();
            
            for (int depth = 0; depth < ilog2ceil(2 * blockSize); ++depth) {
                kernUpSweep<<<blocksPerGrid, blockSize>>>(n_padded, depth, dev_data);

                blockSize /= 2; // need fewer threads each iteration
                cudaDeviceSynchronize();
            }

            // Keep blocks per grid constant, so we can handle arbitrarily sized arrays.
            // But grow blockSize on each iteration since we need more threads on each depth layer.
            blockSize = 1;
            for (int depth = ilog2ceil(2 * blockSize) - 1; depth >= 0; --depth) {
                kernDownSweep<<<blocksPerGrid, blockSize>>>(n_padded, depth, dev_data, stored_sums);

                blockSize *= 2; // need more threads each iteration
                cudaDeviceSynchronize();
            }

            // If the array didn't fit within a single block, we need to collect the individual block scan results, 
            // put them in an array, and scan that array. Then add the twice-scanned array as increments back to the original results.
            //
            // This needs to be done recursively to handle arbitrarily large arrays.
            if (n_padded > 2 * MAX_BLOCK_SIZE) {
                // TODO - partial scans are all exclusive... need to be inclusive for this step

                // Scatter the sums from the previous scan operation
                int* sum_data;
                cudaMalloc((void**) &sum_data, blocksPerGrid.x * sizeof(int));
                int sumBlockSize = 128;
                dim3 sumBlocksPerGrid = (blocksPerGrid.x + sumBlockSize - 1) / sumBlockSize;
                kernBlockSums<<<sumBlocksPerGrid, sumBlockSize>>>(blocksPerGrid.x, blockSize, dev_data, sum_data, stored_sums);
                cudaDeviceSynchronize();

                // (Recursively) scan the summed blocks array
                // Can use sum_data as both the input and output pointers for the scan. No issue writing over it.
                scan(blocksPerGrid.x, sum_data, sum_data);

                // Finally, add scanned sum values back to the original dev_data
                // In original scan, each thread handled 2 elements. In this step, each handles one, so we need 2x the blocks.
                dim3 kernBlocksPerGrid = 2 * blocksPerGrid.x; 
                kernIncrement<<<kernBlocksPerGrid, blockSize>>>(n_padded, dev_data, sum_data);
                cudaDeviceSynchronize();

                cudaFree(sum_data);
            }

            // TODO - the recursive scan  is going to trigger multiple timer calls... create wrapper function for timing.
            // Actually I think best bet is to modify timer to not restart if it's already been started? But then what about end... wrapper function for that?
            // timer().endGpuTimer();
            cudaFree(stored_sums);
            cudaMemcpy(odata, dev_data, n_padded * sizeof(int), cudaMemcpyDeviceToHost);
            cudaFree(dev_data);
        }

        __global__ void kernUpSweep(int n, int depth, int* dev_data) {
            int threadId = threadIdx.x; 
            if (threadId + (blockDim.x * blockIdx.x) >= n/2) return;

            int twoToDepthPlusOne = (1 << (depth + 1));
            int twoToDepth = (1 << depth);
            // Since each block is a self contained scan, we calculate these indices w.r.t the local block thread index.
            // But then we offset by (blockDim.x * blockIdx.x) because the dev_data is for ALL blocks, so we need to access the right part.
            int leftChildIdx = (threadId * twoToDepthPlusOne) + twoToDepth - 1 + (blockDim.x * blockIdx.x);
            int rightChildIdx = (threadId * twoToDepthPlusOne) + twoToDepthPlusOne - 1 + (blockDim.x * blockIdx.x);

            dev_data[rightChildIdx] += dev_data[leftChildIdx];
        }

        __global__ void kernDownSweep(int n, int depth, int* dev_data, int* stored_sums) {
            int threadId = threadIdx.x; 
            if (threadId + (blockDim.x * blockIdx.x) >= n/2) return;

            int twoToDepthPlusOne = (1 << (depth + 1));
            int twoToDepth = (1 << depth);
            int blockLeftChildIdx = (threadId * twoToDepthPlusOne) + twoToDepth - 1;
            int globalLeftChildIdx = blockLeftChildIdx + (blockDim.x * blockIdx.x);
            int blockRightChildIdx = (threadId * twoToDepthPlusOne) + twoToDepthPlusOne - 1;
            int globalRightChildIdx = blockRightChildIdx + + (blockDim.x * blockIdx.x);

            if (blockRightChildIdx == (blockDim.x - 1)) {
                stored_sums[blockIdx.x] = dev_data[blockRightChildIdx]; // need this for later kernel
                dev_data[blockRightChildIdx] = 0; // zero out last element in block
            }

            int leftVal = dev_data[globalLeftChildIdx];
            dev_data[globalLeftChildIdx] = dev_data[globalRightChildIdx];
            dev_data[globalLeftChildIdx] += leftVal;
        }

        /**
         * Kernel to scatter sums from individual blocks into a sum_data array.
         * In most use cases, the sum_data array is probably small and this would be faster on the CPU, but for REALLY big input arrays,
         * it might be worth it to do this step on the GPU.
         * 
         * Here, n is the number of blocks used in the upsweep / downsweep steps above.
         */
        __global__ void kernBlockSums(int n, int stride, const int* dev_data, int* sum_data, const int* stored_sums) {
            int threadId = threadIdx.x + (blockDim.x * blockIdx.x);
            if (threadId > n) return;

            // Note: since the partial scans on dev_data were exclusive, we need to do one more addition to get the inclusive scan amount.
            sum_data[threadId] = dev_data[(threadId * stride) + (stride - 1)] + stored_sums[threadId]; 
        }

        /**
         * Kernel to add the scanned block sums back to the original array.
         * 
         * n here is the number of elements in the original input arary.
         */
        __global__ void kernIncrement(int n, int* dev_data, int* sum_data) {
            int threadId = threadIdx.x + (blockIdx.x * blockDim.x);
            if (threadId > n) return;

            // Divide blockIdx.x by 2 because we're using 2x the blocks for this step compared to the original scan.
            dev_data[threadId] += sum_data[blockIdx.x / 2];
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

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

        const int MAX_BLOCK_SIZE = 1024; // keep this as a power of 2

        void scan(int n_padded, int* dev_data) {
            int blockSize = std::min((n_padded / 2), MAX_BLOCK_SIZE);
            dim3 blocksPerGrid = ((n_padded / 2) + blockSize - 1) / blockSize;

            int* stored_sums; // temp array used to store last entry per block during upsweep. See kernZeroEntries and kernIncrement for use info.
            cudaMalloc((void**)&stored_sums, blocksPerGrid.x * sizeof(int));

            kernScan<<<blocksPerGrid, blockSize, 2 * sizeof(int) * blockSize>>>(n_padded, ilog2ceil(2 * blockSize), dev_data, stored_sums);
            cudaDeviceSynchronize();

            // If the array didn't fit within a single block, we need to collect the individual block scan results, 
            // put them in an array, and scan that array. Then add the twice-scanned array as increments back to the original results.
            //
            // This needs to be done recursively to handle arbitrarily large arrays.
            if (n_padded > 2 * blockSize) {
                // (Recursively) scan the summed blocks array
                // Can use sum_data as both the input and output pointers for the scan. No issue writing over it.
                scan(blocksPerGrid.x, stored_sums);

                // Finally, add scanned sum values back to the original dev_data
                // In original scan, each thread handled 2 elements. In this step, each handles one, so we need 2x the blocks.
                dim3 kernBlocksPerGrid = 2 * blocksPerGrid.x;
                kernIncrement<<<kernBlocksPerGrid, blockSize>>>(n_padded, dev_data, stored_sums);
                cudaDeviceSynchronize();
            }

            cudaFree(stored_sums);
        }

        /**
         * Wrapper method (to facilitate gpu timing and allocating things)
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

        /**
         * n is the size of dev_data
         */
        __global__ void kernScan(int n, int numLevels, int* dev_data, int* stored_sums) {
            if (threadIdx.x + (blockDim.x * blockIdx.x) >= n/2) return;

            extern __shared__ int s_dev_data[];

            // Put the right and left children into shared memory.
            // Index from dev_data based on *global* position, but put into shared memory as local position (w.r.t. this block)
            s_dev_data[2 * threadIdx.x + 1] = 
                dev_data[2 * threadIdx.x + 1 + (2 * blockDim.x * blockIdx.x)];

            s_dev_data[2 * threadIdx.x] =
                dev_data[2 * threadIdx.x + (2 * blockDim.x * blockIdx.x)];

            for (int depth = 0; depth < numLevels; ++depth) {
                __syncthreads();
                // Make sure the local right-child index this thread will access is in bounds of this block.
                if ((threadIdx.x * (1 << (depth + 1))) + (1 << (depth + 1)) - 1 >= 2 * blockDim.x) continue;
                
                s_dev_data[(threadIdx.x * (1 << (depth + 1))) + (1 << (depth + 1)) - 1] +=
                    s_dev_data[(threadIdx.x * (1 << (depth + 1))) + (1 << depth) - 1];
            }
            __syncthreads();

            // Save off the last entry of s_dev_data to a temporary stored_sums buffer. This temp buffer is used if our initial
            // input data is too large to be scanned in a single block.
            // Then zero out the last entry for the downsweep.
            if (threadIdx.x == 0) {
                stored_sums[blockIdx.x] = s_dev_data[2 * blockDim.x - 1];
                s_dev_data[2 * blockDim.x - 1] = 0;
            }
            
            for (int depth = numLevels - 1; depth >= 1; --depth) {
                __syncthreads();
                // Make sure the local right-child index this thread will access is in bounds of this block.
                if ((threadIdx.x * (1 << (depth + 1))) + (1 << (depth + 1)) - 1 >= 2 * blockDim.x) continue;

                int leftVal = s_dev_data[(threadIdx.x * (1 << (depth + 1))) + (1 << depth) - 1];
                s_dev_data[(threadIdx.x * (1 << (depth + 1))) + (1 << depth) - 1] = 
                    s_dev_data[(threadIdx.x * (1 << (depth + 1))) + (1 << (depth + 1)) - 1];
                
                s_dev_data[(threadIdx.x * (1 << (depth + 1))) + (1 << (depth + 1)) - 1] += leftVal;
            }

            __syncthreads();
            // On the last iteration, depth = 0, we write to global memory.
            dev_data[2 * threadIdx.x + (2 * blockDim.x * blockIdx.x)] = 
                s_dev_data[2 * threadIdx.x + 1];
            
            dev_data[2 * threadIdx.x + 1 + (2 * blockDim.x * blockIdx.x)] = 
                (s_dev_data[2 * threadIdx.x] + s_dev_data[2 * threadIdx.x + 1]);
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
            int sum_data_idx = (long) gridDim.x * threadId / (2 * n); // note: integer division here

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
            int n_padded = pow(2, ilog2ceil(n)); // pad array to nearest power of two
            int* trueFalseArray, *dev_idata, *dev_odata;
            cudaMalloc((void**)&trueFalseArray, n_padded * sizeof(int));
            cudaMalloc((void**)&dev_idata, n * sizeof(int));
            cudaMalloc((void**)&dev_odata, n * sizeof(int)); // allocate for worst-case scenario
            cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);

            if (n < n_padded) {
                // Note that if we wanted to *retain* 0s in the compacted array, we'd need to pad
                // trueFalseArray with a different value here.
                cudaMemset(trueFalseArray + n, 0, (n_padded - n) * sizeof(int));
            }

            timer().startGpuTimer();
            
            int threadsPerBlock = 128;
            dim3 blocksPerGrid = ((n + threadsPerBlock - 1) / threadsPerBlock);
            StreamCompaction::Common::kernMapToBoolean<<<blocksPerGrid, threadsPerBlock>>>(n, trueFalseArray, dev_idata);
            cudaDeviceSynchronize();

            scan(n_padded, trueFalseArray); // scan happens in-place, so trueFalseArray is now scanned

            StreamCompaction::Common::kernScatter<<<blocksPerGrid, threadsPerBlock>>>(n, dev_odata, dev_idata, trueFalseArray);
            cudaDeviceSynchronize();

            timer().endGpuTimer();

            int compactArraySize;
            cudaMemcpy(&compactArraySize, trueFalseArray + n - 1, sizeof(int), cudaMemcpyDeviceToHost);
            compactArraySize += (idata[n - 1] != 0); // necessary because scan was exclusive

            cudaMemcpy(odata, dev_odata, compactArraySize * sizeof(int), cudaMemcpyDeviceToHost);
            cudaFree(dev_odata);
            cudaFree(dev_idata);
            cudaFree(trueFalseArray);
            return compactArraySize;
        }
    }
}

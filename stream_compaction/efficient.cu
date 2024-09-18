#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"
#include "device_launch_parameters.h"
#include <stdio.h>

#ifndef __CUDACC__  
#define __CUDACC__
#endif

#define NUM_BANKS 16
#define LOG_NUM_BANKS 4
#define CONFLICT_FREE_OFFSET(n) ((n) >> LOG_NUM_BANKS)


namespace StreamCompaction {
    namespace Efficient {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }
        
        // helper function to get next power of two (for host)
        __host__ int h_nextPowerOfTwo(int n) {
            int power = 1;
            while (power < n)
                power <<= 1;
            return power;
        }

        // helper function to get next power of two (for device)
        __device__ int d_nextPowerOfTwo(int x) {
            if (x == 0) {
                return 1;
            }

            x--;

            x |= x >> 1;
            x |= x >> 2;
            x |= x >> 4;
            x |= x >> 8;
            x |= x >> 16;

            return x + 1;
        }

        // do scan from GPU Gems
        __global__ void scan_ker(int* g_odata, const int* g_idata, int* g_block_sums, int n) {
            extern __shared__ int temp[]; // allocated on invocation
            int thid = threadIdx.x;

            int blockOffset = blockIdx.x * blockDim.x * 2;
            int ai = thid;
            int bi = thid + blockDim.x;

            // number of elements to process in this block
            int n_block = 2 * blockDim.x;

            // next power of two greater or equal to n_block
            int n_shared = n_block;
            int bankOffsetA = CONFLICT_FREE_OFFSET(ai);
            int bankOffsetB = CONFLICT_FREE_OFFSET(bi);

            // load input into shared memory with padding
            if (blockOffset + ai < n)
                temp[ai + bankOffsetA] = g_idata[blockOffset + ai];
            else
                temp[ai + bankOffsetA] = 0;
            if (blockOffset + bi < n)
                temp[bi + bankOffsetB] = g_idata[blockOffset + bi];
            else
                temp[bi + bankOffsetB] = 0;

            // build sum in place up the tree
            int offset = 1;
            for (int d = n_shared >> 1; d > 0; d >>= 1)
            {
                __syncthreads();
                if (thid < d)
                {
                    int ai = offset * (2 * thid + 1) - 1;
                    int bi = offset * (2 * thid + 2) - 1;

                    int bankOffsetA = CONFLICT_FREE_OFFSET(ai);
                    int bankOffsetB = CONFLICT_FREE_OFFSET(bi);

                    temp[bi + bankOffsetB] += temp[ai + bankOffsetA];
                }
                offset <<= 1;
            }

            // clear the last element
            if (thid == 0) {
                if (g_block_sums != NULL)
                    g_block_sums[blockIdx.x] = temp[n_shared - 1 + CONFLICT_FREE_OFFSET(n_shared - 1)];
                temp[n_shared - 1 + CONFLICT_FREE_OFFSET(n_shared - 1)] = 0;
            }

            // traverse down tree & build scan
            for (int d = 1; d < n_shared; d <<= 1)
            {
                offset >>= 1;
                __syncthreads();
                if (thid < d)
                {
                    int ai = offset * (2 * thid + 1) - 1;
                    int bi = offset * (2 * thid + 2) - 1;

                    int bankOffsetA = CONFLICT_FREE_OFFSET(ai);
                    int bankOffsetB = CONFLICT_FREE_OFFSET(bi);

                    int t = temp[ai + bankOffsetA];
                    temp[ai + bankOffsetA] = temp[bi + bankOffsetB];
                    temp[bi + bankOffsetB] += t;
                }
            }
            __syncthreads();

            // write results to global memory
            if (blockOffset + ai < n)
                g_odata[blockOffset + ai] = temp[ai + bankOffsetA];
            if (blockOffset + bi < n)
                g_odata[blockOffset + bi] = temp[bi + bankOffsetB];
        }

        // kernel to add the scanned block sums to each block
        __global__ void add_scanned_block_sums(int* g_data, const int* g_block_sums, int n) {
            int index = threadIdx.x + blockIdx.x * blockDim.x * 2;
            int offset = blockIdx.x;

            if (offset == 0) return; // skip first block

            int addValue = g_block_sums[offset];

            if (index < n)
                g_data[index] += addValue;
            if (index + blockDim.x < n)
                g_data[index + blockDim.x] += addValue;
        }

        // scan function
        void scanRecursive(int n, int* d_odata, const int* d_idata) {
            // base case
            if (n <= 1024) {
                int threadsPerBlock = (n + 1) / 2;
                int sharedMemSize = h_nextPowerOfTwo(n) * sizeof(int);

                scan_ker<<<1, threadsPerBlock, sharedMemSize>>>(d_odata, d_idata, NULL, n);
                cudaDeviceSynchronize();
                checkCUDAError("scan_ker base case kernel execution");
                return;
            }

            // determine block and grid sizes
            int threadsPerBlock = 512;
            int elementsPerBlock = threadsPerBlock * 2;
            int numBlocks = (n + elementsPerBlock - 1) / elementsPerBlock;

            // allocate memory for block sums
            int* d_block_sums;
            cudaMalloc((void**)&d_block_sums, numBlocks * sizeof(int));
            checkCUDAError("cudaMalloc d_block_sums");

            // shared memory size per block
            int sharedMemSize = 2 * threadsPerBlock * sizeof(int);

            // launch the scan kernel
            scan_ker <<<numBlocks, threadsPerBlock, sharedMemSize>>>(d_odata, d_idata, d_block_sums, n);
            cudaDeviceSynchronize();
            checkCUDAError("scan_ker kernel execution");

            // if there is more than one block, we need to scan the block sums and add them to the data
            if (numBlocks > 1) {
                // allocate memory for scanned block sums
                int* d_scanned_block_sums;
                cudaMalloc((void**)&d_scanned_block_sums, numBlocks * sizeof(int));
                checkCUDAError("cudaMalloc d_scanned_block_sums");

                // recursively call scanRecursive on the block sums array
                scanRecursive(numBlocks, d_scanned_block_sums, d_block_sums);

                // launch kernel to add scanned block sums to data
                add_scanned_block_sums<<<numBlocks, threadsPerBlock>>>(d_odata, d_scanned_block_sums, n);
                cudaDeviceSynchronize();
                checkCUDAError("add_scanned_block_sums kernel execution");

                cudaFree(d_scanned_block_sums);
            }

            cudaFree(d_block_sums);
        }

        // scan (has timer)
        void scan(int n, int* odata, const int* idata) {
            // allocate device memory
            int* d_idata, * d_odata;
            cudaMalloc((void**)&d_idata, n * sizeof(int));
            checkCUDAError("cudaMalloc d_idata");
            cudaMalloc((void**)&d_odata, n * sizeof(int));
            checkCUDAError("cudaMalloc d_odata");

            // copy input data to device
            cudaMemcpy(d_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            checkCUDAError("cudaMemcpy to d_idata");

            timer().startGpuTimer();

            // call the recursive scan function
            scanRecursive(n, d_odata, d_idata);

            timer().endGpuTimer();

            // copy result back to host
            cudaMemcpy(odata, d_odata, n * sizeof(int), cudaMemcpyDeviceToHost);
            checkCUDAError("cudaMemcpy to odata");

            // free device memory
            cudaFree(d_idata);
            cudaFree(d_odata);
        }
  
        // same as above, but does not call the GPU timer, to be used within a larger call in stream compact (efficient)
        void scan_no_timer(int n, int* odata, const int* idata) {
            // allocate device memory
            int* d_idata, * d_odata;
            cudaMalloc((void**)&d_idata, n * sizeof(int));
            checkCUDAError("cudaMalloc d_idata");
            cudaMalloc((void**)&d_odata, n * sizeof(int));
            checkCUDAError("cudaMalloc d_odata");

            // copy input data to device
            cudaMemcpy(d_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            checkCUDAError("cudaMemcpy to d_idata");


            // call the recursive scan function
            scanRecursive(n, d_odata, d_idata);


            // copy result back to host
            cudaMemcpy(odata, d_odata, n * sizeof(int), cudaMemcpyDeviceToHost);
            checkCUDAError("cudaMemcpy to odata");

            // free device memory
            cudaFree(d_idata);
            cudaFree(d_odata);
        }

        // compact
        int compact(int n, int* odata, const int* idata) {
            // allocate memory on the device
            int* d_idata, * d_bools, * d_indices, * d_odata;
            cudaMalloc((void**)&d_idata, n * sizeof(int));
            cudaMalloc((void**)&d_bools, n * sizeof(int));
            cudaMalloc((void**)&d_indices, n * sizeof(int));
            cudaMalloc((void**)&d_odata, n * sizeof(int));

            // copy input data to device
            cudaMemcpy(d_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);

            int blockSize = 1024;
            int gridSize = (n + blockSize - 1) / blockSize;  // Ensure all elements are covered

            // start timing after memory allocations and copies
            timer().startGpuTimer();

            // map input data to boolean (1 for non-zero, 0 for zero)
            StreamCompaction::Common::kernMapToBoolean<<<gridSize, blockSize>>>(n, d_bools, d_idata);
            cudaDeviceSynchronize();

            // perform an exclusive prefix sum (scan) on the boolean array
            scan_no_timer(n, d_indices, d_bools);
            cudaDeviceSynchronize();

            // scatter non-zero elements from idata to odata based on the scan results
            StreamCompaction::Common::kernScatter<<<gridSize, blockSize>>>(n, d_odata, d_idata, d_bools, d_indices);
            cudaDeviceSynchronize();

            // end timing before any device-to-host memory transfers
            timer().endGpuTimer();

            // retrieve the number of valid (non-zero) elements
            int numValidElements;
            cudaMemcpy(&numValidElements, &d_indices[n - 1], sizeof(int), cudaMemcpyDeviceToHost);

            // check if the last element is valid (if bools[n - 1] is 1, add 1 to numValidElements)
            int lastBool;
            cudaMemcpy(&lastBool, &d_bools[n - 1], sizeof(int), cudaMemcpyDeviceToHost);

            if (lastBool == 1) {
                numValidElements += 1;
            }

            // copy the compacted data to the output array on the host
            cudaMemcpy(odata, d_odata, numValidElements * sizeof(int), cudaMemcpyDeviceToHost);

            // free device memory
            cudaFree(d_idata);
            cudaFree(d_bools);
            cudaFree(d_indices);
            cudaFree(d_odata);

            // return the number of elements remaining after compaction
            return numValidElements;
        }



    }
}

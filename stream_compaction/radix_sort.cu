#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include "common.h"
#include "radix_sort.h"
#include "efficient_optimized.h"
#include <device_functions.h>

namespace StreamCompaction {
	namespace RadixSort
	{
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

#pragma region GpuScan
        __global__
            void gpu_add_block_sums(int* const d_out,
                const int* const d_in,
                int* const d_block_sums,
                const size_t numElems)
        {
            //unsigned int glbl_t_idx = blockDim.x * blockIdx.x + threadIdx.x;
            unsigned int d_block_sum_val = d_block_sums[blockIdx.x];

            //unsigned int d_in_val_0 = 0;
            //unsigned int d_in_val_1 = 0;

            // Simple implementation's performance is not significantly (if at all)
            //  better than previous verbose implementation
            unsigned int cpy_idx = 2 * blockIdx.x * blockDim.x + threadIdx.x;
            if (cpy_idx < numElems)
            {
                d_out[cpy_idx] = d_in[cpy_idx] + d_block_sum_val;
                if (cpy_idx + blockDim.x < numElems)
                    d_out[cpy_idx + blockDim.x] = d_in[cpy_idx + blockDim.x] + d_block_sum_val;
            }
        }

        // Modified version of Mark Harris' implementation of the Blelloch scan
        // according to https://www.mimuw.edu.pl/~ps209291/kgkp/slides/scan.pdf
        __global__
            void gpu_prescan(int* const d_out,
                int* const d_in,
                int* const d_block_sums,
                const unsigned int len,
                const unsigned int shmem_sz,
                const unsigned int max_elems_per_block)
        {
            // Allocated on invocation
            extern __shared__ unsigned int s_out[];

            int thid = threadIdx.x;
            int ai = thid;
            int bi = thid + blockDim.x;

            // Zero out the shared memory
            // Helpful especially when input size is not power of two
            s_out[thid] = 0;
            s_out[thid + blockDim.x] = 0;
            // If CONFLICT_FREE_OFFSET is used, shared memory
            //  must be a few more than 2 * blockDim.x
            if (thid + max_elems_per_block < shmem_sz)
                s_out[thid + max_elems_per_block] = 0;

            __syncthreads();

            // Copy d_in to shared memory
            // Note that d_in's elements are scattered into shared memory
            //  in light of avoiding bank conflicts
            unsigned int cpy_idx = max_elems_per_block * blockIdx.x + threadIdx.x;
            if (cpy_idx < len)
            {
                s_out[ai + CONFLICT_FREE_OFFSET(ai)] = d_in[cpy_idx];
                if (cpy_idx + blockDim.x < len)
                    s_out[bi + CONFLICT_FREE_OFFSET(bi)] = d_in[cpy_idx + blockDim.x];
            }

            // For both upsweep and downsweep:
            // Sequential indices with conflict free padding
            //  Amount of padding = target index / num banks
            //  This "shifts" the target indices by one every multiple
            //   of the num banks
            // offset controls the stride and starting index of 
            //  target elems at every iteration
            // d just controls which threads are active
            // Sweeps are pivoted on the last element of shared memory

            // Upsweep/Reduce step
            int offset = 1;
            for (int d = max_elems_per_block >> 1; d > 0; d >>= 1)
            {
                __syncthreads();

                if (thid < d)
                {
                    int ai = offset * ((thid << 1) + 1) - 1;
                    int bi = offset * ((thid << 1) + 2) - 1;
                    ai += CONFLICT_FREE_OFFSET(ai);
                    bi += CONFLICT_FREE_OFFSET(bi);

                    s_out[bi] += s_out[ai];
                }
                offset <<= 1;
            }

            // Save the total sum on the global block sums array
            // Then clear the last element on the shared memory
            if (thid == 0)
            {
                d_block_sums[blockIdx.x] = s_out[max_elems_per_block - 1
                    + CONFLICT_FREE_OFFSET(max_elems_per_block - 1)];
                s_out[max_elems_per_block - 1
                    + CONFLICT_FREE_OFFSET(max_elems_per_block - 1)] = 0;
            }

            // Downsweep step
            for (int d = 1; d < max_elems_per_block; d <<= 1)
            {
                offset >>= 1;
                __syncthreads();

                if (thid < d)
                {
                    int ai = offset * ((thid << 1) + 1) - 1;
                    int bi = offset * ((thid << 1) + 2) - 1;
                    ai += CONFLICT_FREE_OFFSET(ai);
                    bi += CONFLICT_FREE_OFFSET(bi);

                    unsigned int temp = s_out[ai];
                    s_out[ai] = s_out[bi];
                    s_out[bi] += temp;
                }
            }
            __syncthreads();

            // Copy contents of shared memory to global memory
            if (cpy_idx < len)
            {
                d_out[cpy_idx] = s_out[ai + CONFLICT_FREE_OFFSET(ai)];
                if (cpy_idx + blockDim.x < len)
                    d_out[cpy_idx + blockDim.x] = s_out[bi + CONFLICT_FREE_OFFSET(bi)];
            }
        }

        void gpuScanOptimized(int* dev_odata,
            int* dev_idata,
            int* d_block_sums,
            int* d_dummy_blocks_sums,
            int n);

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int* odata, const int* idata, bool startTimer, bool isHost)
        {
            int* dev_odata;
            int* dev_idata;

            unsigned int blockSize = MAX_BLOCK_SIZE / 2;
            unsigned int maxElemsPerBlock = 2 * blockSize;

            cudaMalloc((void**)&dev_odata, n * sizeof(int));
            checkCUDAError("cudaMalloc dev_odata failed!");
            cudaMemset(dev_odata, 0, n * sizeof(unsigned int));
            checkCUDAError("cudaMemset dev_odata failed!");

            cudaMalloc((void**)&dev_idata, n * sizeof(int));
            checkCUDAError("cudaMalloc dev_idata failed!");
            cudaMemset(dev_idata, 0, n * sizeof(unsigned int));
            checkCUDAError("cudaMemset dev_idata failed!");
            cudaMemcpy(dev_idata, idata, n * sizeof(int), isHost ? cudaMemcpyHostToDevice : cudaMemcpyDeviceToDevice);
            checkCUDAError("cudaMemcpy dev_idata failed!");

            unsigned int gridSize = (n + maxElemsPerBlock - 1) / maxElemsPerBlock;

            // Conflict free padding requires that shared memory be more than 2 * block_sz
            unsigned int shmemSize = maxElemsPerBlock + ((maxElemsPerBlock - 1) >> LOG_NUM_BANKS);

            // Allocate memory for array of total sums produced by each block
            // Array length must be the same as number of blocks
            int* d_block_sums;
            cudaMalloc(&d_block_sums, sizeof(unsigned int) * gridSize);
            checkCUDAError("cudaMalloc d_block_sums failed!");
            cudaMemset(d_block_sums, 0, sizeof(unsigned int) * gridSize);
            checkCUDAError("cudaMemset d_block_sums failed!");

            int gridSizeSums = (gridSize + maxElemsPerBlock - 1) / maxElemsPerBlock;

            int* d_dummy_blocks_sums;
            cudaMalloc(&d_dummy_blocks_sums, sizeof(unsigned int) * gridSizeSums);
            checkCUDAError("cudaMalloc d_dummy_blocks_sums failed!");
            cudaMemset(d_dummy_blocks_sums, 0, sizeof(unsigned int) * gridSizeSums);
            checkCUDAError("cudaMemset d_dummy_blocks_sums failed!");

            if (startTimer)
            {
                timer().startGpuTimer();
            }

            gpuScanOptimized(dev_odata, dev_idata, d_block_sums, d_dummy_blocks_sums, n);

            if (startTimer)
            {
                timer().endGpuTimer();
            }

            cudaMemcpy(odata, dev_odata, n * sizeof(int), cudaMemcpyDeviceToDevice);

            cudaFree(dev_odata);
            checkCUDAError("cudaFree dev_odata failed!");
            cudaFree(dev_idata);
            checkCUDAError("cudaFree dev_idata failed!");
            cudaFree(d_block_sums);
            checkCUDAError("cudaFree d_block_sums failed!");
            cudaFree(d_dummy_blocks_sums);
            checkCUDAError("cudaFree d_dummy_blocks_sums failed!");
        }

        void gpuScanOptimized(int* dev_odata,
            int* dev_idata,
            int* d_block_sums,
            int* d_dummy_blocks_sums,
            int n)
        {
            unsigned int blockSize = MAX_BLOCK_SIZE / 2;
            unsigned int maxElemsPerBlock = 2 * blockSize;
            unsigned int gridSize = (n + maxElemsPerBlock - 1) / maxElemsPerBlock;
            // Conflict free padding requires that shared memory be more than 2 * block_sz
            unsigned int shmemSize = maxElemsPerBlock + ((maxElemsPerBlock - 1) >> LOG_NUM_BANKS);

            gpu_prescan << <gridSize, blockSize, sizeof(unsigned int)* shmemSize >> > (
                dev_odata,
                dev_idata,
                d_block_sums,
                n,
                shmemSize,
                maxElemsPerBlock);

            // compute prefix sum of sums array
            // sums:
            // [data[0] + .... + data[511], data[512] + ... + data[1023], ....]
            // ---->
            // [data[0] + .... + data[511], data[0] + ... + data[1023], ....]
            if (gridSize <= maxElemsPerBlock)
            {
                gpu_prescan << <1, blockSize, sizeof(unsigned int)* shmemSize >> > (
                    d_block_sums,
                    d_block_sums,
                    d_dummy_blocks_sums,
                    gridSize,
                    shmemSize,
                    maxElemsPerBlock);
            }
            else
            {
                int* d_in_block_sums;
                cudaMalloc(&d_in_block_sums, sizeof(unsigned int) * gridSize);
                checkCUDAError("cudaMalloc d_in_block_sums failed!");
                cudaMemcpy(d_in_block_sums, d_block_sums, sizeof(unsigned int) * gridSize, cudaMemcpyDeviceToDevice);
                checkCUDAError("cudaMemcpy d_in_block_sums failed!");

                scan(gridSize, d_block_sums, d_in_block_sums, false, false);

                cudaFree(d_in_block_sums);
                checkCUDAError("cudaFree d_in_block_sums failed!");
            }

            gpu_add_block_sums << <gridSize, blockSize >> > (dev_odata, dev_odata, d_block_sums, n);
        }
#pragma endregion

        __global__ void kernMapToBool(int i, int n,
                                       int* bitBuffer,
                                       int* dev_idata, 
                                       int* skip)
        {
            int index = threadIdx.x + (blockIdx.x * blockDim.x);
            if (index >= n) return;
            int num = dev_idata[index];
            int temp = (num & (1 << i)) >> i;
            if (i == 0 || temp != ((num & (1 << (i - 1))) >> (i - 1)))
            {
                *skip = 1;
            }
            bitBuffer[index] = 1 - temp;
        }

        __global__ void kernScatter(int i, int n, int startIdx,
                               int* scanBuffer,
                               int* dev_idata,
                               int* dev_odata)
        {
            int index = threadIdx.x + (blockIdx.x * blockDim.x);
            if (index >= n) return;
            int num = dev_idata[index];
            bool flag = (num & (1 << i)) >> i;
            int t = index - scanBuffer[index] + startIdx;
            dev_odata[flag ? t : scanBuffer[index]] = num;
        }

        __global__ void kernalCheckStop(int n, const int* idata, int* stop)
        {
            int index = threadIdx.x + (blockDim.x * blockIdx.x);
            if (index >= n - 1) return;

            if (idata[index] > idata[index + 1]) (*stop) = 1;
        }

        void radixSort(int i, int n,
                       int* bitBuffer,
                       int* dev_idata,
                       int* dev_odata,
                       int* d_block_sums,
                       int* d_dummy_blocks_sums,
                       int* dev_num)
        {
            int gridSize = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

            cudaMemset(dev_num, 0, sizeof(int));
            kernMapToBool << < gridSize, BLOCK_SIZE >> > (i, n, bitBuffer, dev_idata, dev_num);

            int last_num;
            cudaMemcpy(&last_num, bitBuffer + n - 1, sizeof(int), cudaMemcpyDeviceToHost);

            gpuScanOptimized(bitBuffer, bitBuffer, d_block_sums, d_dummy_blocks_sums, n);

            int start_index;
            cudaMemcpy(&start_index, bitBuffer + n - 1, sizeof(int), cudaMemcpyDeviceToHost);
            start_index += last_num;

            // for debug
            //int* odata = (int*)malloc(n * sizeof(int));
            //cudaMemcpy(odata, bitBuffer, n * sizeof(int), cudaMemcpyDeviceToHost);
            //for (int i = 0; i < n; ++i)
            //{
            //    std::cout << odata[i] << ", ";
            //}
            //std::cout << std::endl;

            kernScatter << < gridSize, BLOCK_SIZE >> > (i, n, start_index, bitBuffer, dev_idata, dev_odata);
            
            // for debug
            //cudaMemcpy(odata, dev_odata, n * sizeof(int), cudaMemcpyDeviceToHost);
            //for (int i = 0; i < n; ++i)
            //{
            //    std::cout << odata[i] << ", ";
            //}
            //std::cout << std::endl;
        }

        void sort(int n, int* odata, const int* idata)
        {
            int bitSizeOfInt = sizeof(int) * 8;

            unsigned int blockSize = MAX_BLOCK_SIZE / 2;
            unsigned int maxElemsPerBlock = 2 * blockSize;
            unsigned int gridSize = (n + maxElemsPerBlock - 1) / maxElemsPerBlock;
            // Conflict free padding requires that shared memory be more than 2 * block_sz
            unsigned int shmemSize = maxElemsPerBlock + ((maxElemsPerBlock - 1) >> LOG_NUM_BANKS);
            int gridSizeSums = (gridSize + maxElemsPerBlock - 1) / maxElemsPerBlock;

            int* bitBuffer;
            int* reverseBitBuffer;
            int* scanBuffer;
            int* dev_odata, * dev_odataTemp;
            int* dev_int;

            cudaMalloc((void**)&dev_odata, n * sizeof(int));
            checkCUDAError("cudaMalloc dev_odata failed!");
            cudaMemcpy(dev_odata, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            checkCUDAError("cudaMemcpy dev_odata failed!");

            cudaMalloc((void**)&dev_odataTemp, n * sizeof(int));
            checkCUDAError("cudaMalloc dev_odataTemp failed!");

            cudaMalloc((void**)&bitBuffer, n * sizeof(int));
            checkCUDAError("cudaMalloc bitBuffer failed!");

            cudaMalloc((void**)&reverseBitBuffer, n * sizeof(int));
            checkCUDAError("cudaMalloc reverseBitBuffer failed!");

            cudaMalloc((void**)&scanBuffer, n * sizeof(int));
            checkCUDAError("cudaMalloc scanBuffer failed!");

            cudaMalloc((void**)&dev_int, sizeof(int));
            checkCUDAError("cudaMalloc dev_int failed!");

            int* d_block_sums;
            cudaMalloc(&d_block_sums, sizeof(unsigned int) * gridSize);
            checkCUDAError("cudaMalloc d_block_sums failed!");
            cudaMemset(d_block_sums, 0, sizeof(unsigned int) * gridSize);
            checkCUDAError("cudaMemset d_block_sums failed!");

            int* d_dummy_blocks_sums;
            cudaMalloc(&d_dummy_blocks_sums, sizeof(unsigned int) * gridSizeSums);
            checkCUDAError("cudaMalloc d_dummy_blocks_sums failed!");
            cudaMemset(d_dummy_blocks_sums, 0, sizeof(unsigned int) * gridSizeSums);
            checkCUDAError("cudaMemset d_dummy_blocks_sums failed!");

            timer().startGpuTimer();
            for (int i = 0; i < 32; ++i)
            {
                cudaMemset(dev_int, 0, sizeof(int));
                kernalCheckStop << < (n + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE >> > (n, dev_odata, dev_int);
                int stop;
                cudaMemcpy(&stop, dev_int, sizeof(int), cudaMemcpyDeviceToHost);
                if (stop == 0) break;

                radixSort(i, n,
                    bitBuffer,
                    dev_odata,
                    dev_odataTemp,
                    d_block_sums,
                    d_dummy_blocks_sums,
                    dev_int);
                std::swap(dev_odata, dev_odataTemp);

                // for debug
                //cudaMemcpy(odata, dev_odata, n * sizeof(int), cudaMemcpyDeviceToHost);
                //for (int i = 0; i < n; ++i)
                //{
                //    std::cout << odata[i] << ", ";
                //}
                //std::cout << std::endl;
            }
            cudaMemcpy(odata, dev_odata, n * sizeof(int), cudaMemcpyDeviceToHost);
            timer().endGpuTimer();
        }
	}
}
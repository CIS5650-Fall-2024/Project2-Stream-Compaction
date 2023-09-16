#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient_sharedmem.h"
#define BLOCK_THREAD_SIZE 1024
#define BLOCK_ARRAY_SIZE (BLOCK_THREAD_SIZE<<1)
#define SHUFFLE_ADDR 1
namespace StreamCompaction {
    namespace EfficientSharedMem {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }
        //Map [0,block_array_size-1] to [0,block_array_size-1]
        __device__ inline int shuffleAddr(int x, int block_array_size)
        {
            return (x ^ (x >> 3 + 5)) & (block_array_size - 1);
        }

        //Assume N >= 2 and N is power of 2 and every element of input is greater or equal to zero
        __global__ void kernScanBlock(int N, int* input, int* output, int* blockOffset)
        {
            int block_array_size = blockDim.x << 1;
            int block_thread_size = block_array_size >> 1;
            int blockStartIdx = blockIdx.x * block_array_size;
            extern __shared__ int localArr[];
            int tid = threadIdx.x, d_plus_1 = 1;
            int idx1 = blockStartIdx + (tid << 1), idx2 = blockStartIdx + (tid << 1) + 1;
#if SHUFFLE_ADDR
            localArr[shuffleAddr(tid << 1, block_array_size)] = idx1 < N ? input[idx1] : 0;
            localArr[shuffleAddr(1 + (tid << 1), block_array_size)] = idx2 < N ? input[idx2] : 0;
#else // !SHUFFLE_ADDR
            localArr[tid << 1] = idx1 < N ? input[idx1] : 0;
            localArr[1 + (tid << 1)] = idx2 < N ? input[idx2] : 0;
#endif
            __syncthreads();
            for (int numActiveThreads = block_thread_size; numActiveThreads > 0; numActiveThreads >>= 1, d_plus_1++)
            {
                if (tid < numActiveThreads)
                {
                    int two_d_plus_1 = 1 << d_plus_1;
                    int two_d = two_d_plus_1 >> 1;
#if SHUFFLE_ADDR
                    localArr[shuffleAddr(tid * two_d_plus_1 + two_d_plus_1 - 1, block_array_size)] += localArr[shuffleAddr(tid * two_d_plus_1 + two_d - 1, block_array_size)];
#else
                    localArr[tid * two_d_plus_1 + two_d_plus_1 - 1] += localArr[tid * two_d_plus_1 + two_d - 1];
#endif 
                }
                __syncthreads();
            }
            if (tid == 0)
            {
#if SHUFFLE_ADDR
                if (blockOffset)
                    blockOffset[blockIdx.x] = localArr[shuffleAddr(block_array_size - 1, block_array_size)];
                localArr[shuffleAddr(block_array_size - 1, block_array_size)] = 0;
#else
                if(blockOffset)
                    blockOffset[blockIdx.x] = localArr[block_array_size - 1];
                localArr[block_array_size - 1] = 0;
#endif
            }
            __syncthreads();
            for (int numActiveThreads = 1; numActiveThreads <= block_thread_size; numActiveThreads <<= 1)
            {
                d_plus_1--;
                if (tid < numActiveThreads)
                {
                    int two_d_plus_1 = 1 << d_plus_1;
                    int two_d = two_d_plus_1 >> 1;
#if SHUFFLE_ADDR
                    int tmp = localArr[shuffleAddr(tid * two_d_plus_1 + two_d - 1, block_array_size)];
                    localArr[shuffleAddr(tid * two_d_plus_1 + two_d - 1, block_array_size)] = localArr[shuffleAddr(tid * two_d_plus_1 + two_d_plus_1 - 1, block_array_size)];
                    localArr[shuffleAddr(tid * two_d_plus_1 + two_d_plus_1 - 1, block_array_size)] += tmp;
#else
                    int tmp = localArr[tid * two_d_plus_1 + two_d - 1];
                    localArr[tid * two_d_plus_1 + two_d - 1] = localArr[tid * two_d_plus_1 + two_d_plus_1 - 1];
                    localArr[tid * two_d_plus_1 + two_d_plus_1 - 1] += tmp;
#endif
                }
                __syncthreads();
            }
            if (idx1 >= N || idx2 >= N) return;
#if SHUFFLE_ADDR
            output[idx1] = localArr[shuffleAddr(tid << 1, block_array_size)];
            output[idx2] = localArr[shuffleAddr(1 + (tid << 1), block_array_size)];
#else
            output[idx1] = localArr[tid << 1];
            output[idx2] = localArr[1 + (tid << 1)];
#endif
        }

        __global__ void kernAddBlockPrefix(int N, int* output, int* blockOffset, int* blockOffsetPrefix)
        {
            int oblock = blockIdx.x + 1;
            int tid = threadIdx.x;
            int oid1 = oblock * BLOCK_ARRAY_SIZE + (tid << 1), oid2 = oblock * BLOCK_ARRAY_SIZE + (tid << 1) + 1;
            if (oid1 >= N || oid2 >= N) return;
            //add each array element to get inclusive scan
            output[oid1] += blockOffset[blockIdx.x] + blockOffsetPrefix[blockIdx.x];
            output[oid2] += blockOffset[blockIdx.x] + blockOffsetPrefix[blockIdx.x];
        }


        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int* odata, const int* idata) {
            // TODO
            int numBlocks = (n + BLOCK_ARRAY_SIZE - 1) / BLOCK_ARRAY_SIZE;
            size_t N = numBlocks * BLOCK_ARRAY_SIZE;
            int log2numblk = ilog2ceil(numBlocks);
            int num_blk_ceil = 1 << log2numblk;
            int* dev1 = nullptr, * dev3 = nullptr, * dev2 = nullptr, * dev4 = nullptr, * dev5 = nullptr;
            int numBlocksNumBlocks;
            cudaMalloc((void**)&dev1, N * sizeof(int));
            cudaMemset(dev1, 0, N * sizeof(int));
            cudaMemcpy(dev1, idata, sizeof(int) * n, cudaMemcpyHostToDevice);
            if (numBlocks > 1)
            {
                cudaMalloc((void**)&dev2, num_blk_ceil * sizeof(int));
                cudaMalloc((void**)&dev3, num_blk_ceil * sizeof(int));
            }
            if (num_blk_ceil > BLOCK_ARRAY_SIZE)
            {
                numBlocksNumBlocks = num_blk_ceil / BLOCK_ARRAY_SIZE;//num_blk_ceil is already pow of two and greater than BLOCK_ARRAY_SIZE
                cudaMalloc((void**)&dev4, numBlocksNumBlocks * sizeof(int));
                cudaMalloc((void**)&dev5, numBlocksNumBlocks * sizeof(int));
            }
            nvtxRangePushA("Work efficient shared mem scan");
            timer().startGpuTimer();
            kernScanBlock << <numBlocks, BLOCK_THREAD_SIZE, BLOCK_ARRAY_SIZE * sizeof(int) >> > (N, dev1, dev1, dev2);
            cudaDeviceSynchronize();
            checkCUDAError("kernScanBlock error");
            if (numBlocks > 1)//Need to merge between blocks
            {
                if (num_blk_ceil <= BLOCK_ARRAY_SIZE)//Merge once
                {
                    kernScanBlock << <1, num_blk_ceil / 2, num_blk_ceil * sizeof(int) >> > (num_blk_ceil, dev2, dev3, nullptr);
                    cudaDeviceSynchronize();
                    checkCUDAError("kernScanBlock error");
                }
                else//Merge twice
                {
                    kernScanBlock << <numBlocksNumBlocks, BLOCK_THREAD_SIZE, BLOCK_ARRAY_SIZE * sizeof(int) >> > (num_blk_ceil, dev2, dev3, dev4);
                    cudaDeviceSynchronize();
                    checkCUDAError("kernScanBlock error");
                    kernScanBlock << <1, numBlocksNumBlocks / 2, numBlocksNumBlocks * sizeof(int) >> > (numBlocksNumBlocks, dev4, dev5, nullptr);
                    cudaDeviceSynchronize();
                    checkCUDAError("kernScanBlock error");
                    kernAddBlockPrefix << < numBlocksNumBlocks - 1, BLOCK_THREAD_SIZE >> > (num_blk_ceil, dev3, dev4, dev5);
                    cudaDeviceSynchronize();
                    checkCUDAError("kernAddBlockPrefix error");
                }
                kernAddBlockPrefix << < numBlocks - 1, BLOCK_THREAD_SIZE >> > (N, dev1, dev2, dev3);
                cudaDeviceSynchronize();
                checkCUDAError("kernAddBlockPrefix error");
            }
            timer().endGpuTimer();
            nvtxRangePop();
            cudaMemcpy(odata, dev1, sizeof(int) * n, cudaMemcpyDeviceToHost);
            cudaFree(dev1);
            if (numBlocks > 1)
            {
                cudaFree(dev2);
                cudaFree(dev3);
            }
            if (num_blk_ceil > BLOCK_ARRAY_SIZE)
            {
                cudaFree(dev4);
                cudaFree(dev5);
            }
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
        int compact(int n, int* odata, const int* idata) {

            
            return -1;
        }
    }
}

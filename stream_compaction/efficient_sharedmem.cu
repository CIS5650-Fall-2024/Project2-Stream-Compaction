#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient_sharedmem.h"
#define BLOCK_THREAD_SIZE 512
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
        //Map [0,block_array_size-1] to [0,block_array_size-1] this function should be surjective
        __device__ inline int shuffleAddr(int x, int block_array_size)
        {
            return x ^ (x >> 3);
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
            localArr[shuffleAddr(tid << 1, block_array_size)] = input[idx1];
            localArr[shuffleAddr(1 + (tid << 1), block_array_size)] = input[idx2];
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
#if SHUFFLE_ADDR
            output[idx1] = localArr[shuffleAddr(tid << 1, block_array_size)];
            output[idx2] = localArr[shuffleAddr(1 + (tid << 1), block_array_size)];
#else
            output[idx1] = localArr[tid << 1];
            output[idx2] = localArr[1 + (tid << 1)];
#endif
        }

        __global__ void kernAddBlockPrefix(int N,int block_array_size, int* output, int* blockOffset, int* blockOffsetPrefix)
        {
            int oblock = blockIdx.x + 1;
            int tid = threadIdx.x;
            int oid1 = oblock * block_array_size + (tid << 1), oid2 = oblock * block_array_size + (tid << 1) + 1;
            //add each array element to get inclusive scan
            output[oid1] += blockOffset[blockIdx.x] + blockOffsetPrefix[blockIdx.x];
            output[oid2] += blockOffset[blockIdx.x] + blockOffsetPrefix[blockIdx.x];
        }

        gpuScanTempBuffer::gpuScanTempBuffer(int n,int block_array_size, const int* idata)
        {
            this->block_array_size = block_array_size;

            int N = n;
            while (N > block_array_size)
            {
                int currNumBlocks = (N + block_array_size - 1) / block_array_size;
                int* dev1 = nullptr, * dev2 = nullptr;
                cudaMalloc((void**)&dev1, currNumBlocks * block_array_size * sizeof(int));
                if(buffers.size())
                    cudaMalloc((void**)&dev2, currNumBlocks * block_array_size * sizeof(int));
                buffers.emplace_back(dev1, dev2);
                numBlocks.emplace_back(currNumBlocks);
                sharedMemSize.emplace_back(block_array_size * sizeof(int));
                blockSizes.emplace_back(block_array_size / 2);
                numWorkloads.emplace_back(currNumBlocks * block_array_size);
                N = currNumBlocks;
            }
            int* dev1 = nullptr, * dev2 = nullptr;
            N = 1 << (ilog2ceil(N));
            cudaMalloc((void**)&dev1, N * sizeof(int));
            if (buffers.size())
                cudaMalloc((void**)&dev2, N * sizeof(int));
            buffers.emplace_back(dev1, dev2);
            numBlocks.emplace_back(1);
            sharedMemSize.emplace_back(N * sizeof(int));
            blockSizes.emplace_back(N / 2);
            numWorkloads.emplace_back(N);

            cudaMemset(buffers[0].first, 0, numBlocks[0] * sharedMemSize[0]);
            if(idata)
                cudaMemcpy(buffers[0].first, idata, n * sizeof(int), cudaMemcpyHostToDevice);

        }

        gpuScanTempBuffer::~gpuScanTempBuffer()
        {
            for (auto& ptr : buffers)
            {
                if(ptr.first)
                    cudaFree(ptr.first);
                if(ptr.second)
                    cudaFree(ptr.second);
            }
        }

        void gpuScanWorkEfficientOptimized(const gpuScanTempBuffer& tmpBuf)
        {
            int nLevels = tmpBuf.buffers.size();
            for (int i = 0; i < nLevels; i++)
            {
                int numBlock = tmpBuf.numBlocks[i];
                int blockSize = tmpBuf.blockSizes[i];
                int memSize = tmpBuf.sharedMemSize[i];
                int numWorkload = tmpBuf.numWorkloads[i];
                int* thisLevelInputPtr = tmpBuf.buffers[i].first;
                int* thisLevelOutputPtr = i == 0 ? thisLevelInputPtr : tmpBuf.buffers[i].second;
                int* nextLevelPtr = i == nLevels - 1 ? 0 : tmpBuf.buffers[i + 1].first;
                kernScanBlock << <numBlock, blockSize, memSize >> > (numWorkload, thisLevelInputPtr, thisLevelOutputPtr, nextLevelPtr);
                checkCUDAError("kernScanBlock error");
            }
            for (int i = nLevels - 2; i >= 0; i--)
            {
                int numBlock = tmpBuf.numBlocks[i];
                int blockSize = tmpBuf.blockSizes[i];
                int numWorkload = tmpBuf.numWorkloads[i];
                int* thisLevelOutputPtr = i == 0 ? tmpBuf.buffers[i].first : tmpBuf.buffers[i].second;
                int memSize = tmpBuf.sharedMemSize[i];
                int* OffsetPtr = tmpBuf.buffers[i + 1].first;
                int* OffsetPrefixPtr = tmpBuf.buffers[i + 1].second;
                kernAddBlockPrefix << < numBlock - 1, blockSize >> > (numWorkload, tmpBuf.block_array_size, thisLevelOutputPtr, OffsetPtr, OffsetPrefixPtr);
                checkCUDAError("kernAddBlockPrefix error");
            }
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int* odata, const int* idata) {
            // TODO
            gpuScanTempBuffer tmpBuf(n, BLOCK_ARRAY_SIZE, idata);
            nvtxRangePushA("Work efficient shared mem scan");
            timer().startGpuTimer();
            gpuScanWorkEfficientOptimized(tmpBuf);
            timer().endGpuTimer();
            nvtxRangePop();
            cudaMemcpy(odata, tmpBuf.buffers[0].first, sizeof(int) * n, cudaMemcpyDeviceToHost);
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
            
            gpuScanTempBuffer tmpBuf(n, BLOCK_ARRAY_SIZE, idata);
            int* dev_tmp,*dev_out;
            int N = tmpBuf.numWorkloads[0];
            cudaMalloc((void**)&dev_out, N * sizeof(int));
            cudaMalloc((void**)&dev_tmp, N * sizeof(int));
            cudaMemcpy(dev_tmp, tmpBuf.buffers[0].first, N * sizeof(int), cudaMemcpyDeviceToDevice);
            timer().startGpuTimer();
            Common::kernMapToBoolean << <(n + BLOCK_THREAD_SIZE - 1) / BLOCK_THREAD_SIZE, BLOCK_THREAD_SIZE >> > (n, tmpBuf.buffers[0].first, dev_tmp);
            checkCUDAError("kernMapToBoolean error");
            gpuScanWorkEfficientOptimized(tmpBuf);
            Common::kernScatter << <(n + BLOCK_THREAD_SIZE - 1) / BLOCK_THREAD_SIZE, BLOCK_THREAD_SIZE >> > (n, tmpBuf.buffers[0].first, dev_tmp, dev_out, true);
            checkCUDAError("kernScatter error");
            timer().endGpuTimer();
            int excnt;
            cudaMemcpy(&excnt, tmpBuf.buffers[0].first + n - 1, sizeof(int), cudaMemcpyDeviceToHost);
            int cnt = excnt + !!(idata[n - 1]);
            cudaMemcpy(odata, dev_out, sizeof(int) * n, cudaMemcpyDeviceToHost);
            cudaFree(dev_tmp);
            cudaFree(dev_out);
            return cnt;
        }
    }
}

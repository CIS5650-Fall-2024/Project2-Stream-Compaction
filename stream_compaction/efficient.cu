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

        __device__ inline int ilog2(int x) {
            int lg = 0;
            while (x >>= 1) {
                ++lg;
            }
            return lg;
        }

        __device__ inline int ilog2ceil(int x) {
            return x == 1 ? 0 : ilog2(x - 1) + 1;
        }

        int nextPow2(int n) {
            return 1 << (::ilog2ceil(n));
        }

        __global__ void reductionKernel(int n, int *odata, int *idata) {
            extern __shared__ int blockSharedMem[];

            int globalThreadIdx = blockIdx.x*blockDim.x + threadIdx.x;
            if (globalThreadIdx >= n) {
                return;
            }

            blockSharedMem[threadIdx.x] = idata[globalThreadIdx];

            __syncthreads();

            for (int d = 0; d < ilog2(n); ++d) {
                if ((threadIdx.x + 1) % (1 << (d+1)) == 0) {
                    int leftChildValue = blockSharedMem[threadIdx.x - (1 << d)];
                    int rightChildValue = blockSharedMem[threadIdx.x];
                    blockSharedMem[threadIdx.x] = leftChildValue + rightChildValue;
                }

                __syncthreads();
            }

            odata[globalThreadIdx] = blockSharedMem[threadIdx.x];
        }

        __global__ void exclusiveScanKernel(int n, int *odata, int *idata, int *blockSumDevice = nullptr) {
            extern __shared__ int blockSharedMem[];

            int actualBlockSize = n < blockDim.x ? n : blockDim.x;

            int globalThreadIdx = blockIdx.x*blockDim.x + threadIdx.x;
            if (globalThreadIdx >= n) {
                blockSharedMem[threadIdx.x] = 0;
            } else {
                blockSharedMem[threadIdx.x] = idata[globalThreadIdx];
            }

            __syncthreads();

            for (int d = 0; d < ilog2ceil(n); ++d) {
                if ((threadIdx.x + 1) % (1 << (d+1)) == 0) {
                    int leftChildValue = blockSharedMem[threadIdx.x - (1 << d)];
                    int rightChildValue = blockSharedMem[threadIdx.x];
                    blockSharedMem[threadIdx.x] = leftChildValue + rightChildValue;
                }

                __syncthreads();
            }

            if (threadIdx.x == actualBlockSize - 1) {
                blockSharedMem[threadIdx.x] = 0;
            }

            __syncthreads();

            for (int d = ilog2ceil(n) - 1; d >= 0; --d) {
                if ((threadIdx.x + 1) % (1 << (d+1)) == 0) {
                    int leftChildIdx = threadIdx.x - (1 << d);
                    int parentIdx = threadIdx.x;
                    int rightChildIdx = parentIdx;
                    int oldLeftChildValue = blockSharedMem[leftChildIdx];
                    blockSharedMem[leftChildIdx] = blockSharedMem[parentIdx];
                    blockSharedMem[rightChildIdx] += oldLeftChildValue;
                }
                __syncthreads();
            }

            odata[globalThreadIdx] = blockSharedMem[threadIdx.x];

            if (blockSumDevice != nullptr && threadIdx.x == actualBlockSize - 1) {
                blockSumDevice[blockIdx.x] = blockSharedMem[actualBlockSize - 1] + idata[globalThreadIdx];
            }
        }

        __global__ void addBlockSumsKernel(int n, int *odata, int *blockSums) {
            int globalThreadIdx = blockIdx.x*blockDim.x + threadIdx.x;
            if (globalThreadIdx >= n) {
                return;
            }

            odata[globalThreadIdx] += blockSums[blockIdx.x];
        }

        void reduce(int n, int *odata, const int *idata) {
            for (int i = 0; i < n; ++i) {
                odata[i] = idata[i];
            }

            for (int d = 0; d < ::ilog2(n); ++d) {
                for (int k = 0; k < n; k += (1 << (d+1))) {
                    odata[k + (1 << (d+1)) - 1] += odata[k + (1 << d) - 1];
                }
            }
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            timer().startGpuTimer();

            int nNextPow2 = nextPow2(n);

            const int blockSize = 256;
            dim3 blockCount = (nNextPow2 + blockSize - 1) / blockSize;
            // TODO: don't need this much shared memory.
            const int blockSharedMemSize = 2 * blockSize * sizeof(int);

            const int blockSumBlockSize = blockCount.x;
            dim3 blockSumBlockCount = (blockCount.x + blockSumBlockSize - 1) / blockSumBlockSize;
            const int blockSumSharedMemSize = 2 * blockSumBlockSize * sizeof(int);

            int *idataDevice = nullptr;
            cudaMalloc(&idataDevice, nNextPow2 * sizeof(int));
            checkCUDAError("failed to malloc idataDevice");

            int *odataDevice = nullptr;
            cudaMalloc(&odataDevice, nNextPow2 * sizeof(int));
            checkCUDAError("failed to malloc odataDevice");

            int *iblockSumDevice = nullptr;
            cudaMalloc(&iblockSumDevice, blockCount.x * sizeof(int));
            checkCUDAError("failed to malloc iblockSumDevice");

            int *oblockSumDevice = nullptr;
            cudaMalloc(&oblockSumDevice, blockCount.x * sizeof(int));
            checkCUDAError("failed to malloc oblockSumDevice");

            // Debug.
            int *odataDeviceDebug = (int *)malloc(n * sizeof(int));
            int *iblockSumDeviceDebug = (int *)malloc(blockCount.x * sizeof(int));
            int* oblockSumDeviceDebug = (int *)malloc(blockCount.x * sizeof(int));
            int* odataDeviceAfterSumDebug = (int*)malloc(n * sizeof(int));
            // Debug.

            cudaMemcpy(idataDevice, idata, n * sizeof(int), ::cudaMemcpyHostToDevice);

            // reductionKernel<<<blockCount, blockSize, blockSharedMemSize>>>(n, odataDevice, idataDevice);

            exclusiveScanKernel<<<blockCount, blockSize, blockSharedMemSize>>>(nNextPow2, odataDevice, idataDevice, iblockSumDevice);

            // Debug.
            cudaMemcpy(odataDeviceDebug, odataDevice, n * sizeof(int), ::cudaMemcpyDeviceToHost);
            cudaMemcpy(iblockSumDeviceDebug, iblockSumDevice, blockCount.x * sizeof(int), ::cudaMemcpyDeviceToHost);
            // Debug.

            exclusiveScanKernel<<<blockSumBlockCount, blockSumBlockSize, blockSumSharedMemSize>>>(blockCount.x, oblockSumDevice, iblockSumDevice);

            // Debug.
            cudaMemcpy(oblockSumDeviceDebug, oblockSumDevice, blockCount.x * sizeof(int), ::cudaMemcpyDeviceToHost);
            // Debug.

            addBlockSumsKernel<<<blockCount, blockSize>>>(nNextPow2, odataDevice, oblockSumDevice);

            cudaMemcpy(odata, odataDevice, n * sizeof(int), ::cudaMemcpyDeviceToHost);

            cudaFree(oblockSumDevice);
            checkCUDAError("failed to free oblockSumDevice");

            cudaFree(iblockSumDevice);
            checkCUDAError("failed to free iblockSumDevice");

            cudaFree(odataDevice);
            checkCUDAError("failed to free odataDevice");

            cudaFree(idataDevice);
            checkCUDAError("failed to free idataDevice");

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
            // TODO
            timer().endGpuTimer();
            return -1;
        }
    }
}

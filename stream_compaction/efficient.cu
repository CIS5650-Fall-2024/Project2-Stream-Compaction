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

        __global__ void exclusiveScanKernel(int n, int *odata, int *idata, int *blockSumDevice = nullptr) {
            // Block-level scan is done in shared memory.
            extern __shared__ int blockSharedMem[];

            int lastThreadIdxInBlock = (n < blockDim.x ? n : blockDim.x) - 1;

            int globalThreadIdx = blockIdx.x*blockDim.x + threadIdx.x;
            if (globalThreadIdx >= n) {
                // Threads that don't correspond to actual elements in the input array pad the
                // buffer with 0s so that the algorithm can proceed as if the input size were
                // a power of 2.
                blockSharedMem[threadIdx.x] = 0;
            } else {
                blockSharedMem[threadIdx.x] = idata[globalThreadIdx];
            }

            __syncthreads();

            // Up-sweep begins.

            for (int d = 0; d < ilog2ceil(n); ++d) {
                if ((threadIdx.x + 1) % (1 << (d+1)) == 0) {
                    int leftChildValue = blockSharedMem[threadIdx.x - (1 << d)];
                    int rightChildValue = blockSharedMem[threadIdx.x];
                    blockSharedMem[threadIdx.x] = leftChildValue + rightChildValue;
                }

                __syncthreads();
            }

            // Down-sweep begins.

            if (threadIdx.x == lastThreadIdxInBlock) {
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

            // blockSumDevice will store the final prefix sums computed by all blocks to 
            // later combine all blocks' results.
            if (blockSumDevice != nullptr && threadIdx.x == lastThreadIdxInBlock) {
                // An exclusive scan doesn't include the last element, so we need to add it.
                blockSumDevice[blockIdx.x] = blockSharedMem[lastThreadIdxInBlock] + idata[globalThreadIdx];
            }
        }

        // Adds the per-block final prefix sums stored in blockSums to the elements of the 
        // corresponding blocks in odata.
        __global__ void addBlockIncrementsKernel(int n, int *odata, int *blockSums) {
            int globalThreadIdx = blockIdx.x*blockDim.x + threadIdx.x;
            if (globalThreadIdx >= n) {
                return;
            }

            odata[globalThreadIdx] += blockSums[blockIdx.x];
        }

        void scanNoTimer(int n, int *odata, const int *idata) {
            // Round up the input size to the next power of 2 to handle non-power-of-2 inputs
            // and inputs smaller than the block size. Arrays will be padded with 0s to fill
            // the extra space. Since the algorithm arranges its computations in a balanced
            // binary tree, it's easier to have it perform extra work with the extra 0s while
            // following a general algorithm, than to have it handle the edge cases introduced
            // by incompatible input sizes.
            //
            // Padding with 0s is done in the kernel.
            int nNextPow2 = 1 << (::ilog2ceil(n));

            // Kernel configuration for the block-level scan and block sum computation.
            const int blockSize = 256;
            dim3 blockCount = (nNextPow2 + blockSize - 1) / blockSize;
            // No double buffering, so just one block-sized shared memory chunk is needed.
            const int blockSharedMemSize = blockSize * sizeof(int);

            // Kernel configuration for the block sum scan (all blocks' final prefix sums).
            const int blockSumBlockSize = blockCount.x;
            dim3 blockSumBlockCount = (blockCount.x + blockSumBlockSize - 1) / blockSumBlockSize;
            // No double buffering, so just one block-sized shared memory chunk is needed.
            const int blockSumSharedMemSize = blockSumBlockSize * sizeof(int);

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

            // If input needs to be padded with 0s, that'll be done in the kernel.
            cudaMemcpy(idataDevice, idata, n * sizeof(int), cudaMemcpyHostToDevice);

            // Per-block exclusive scan of the original input. iblockSumDevice will store the
            // final prefix sums computed by each block.
            exclusiveScanKernel<<<blockCount, blockSize, blockSharedMemSize>>>(nNextPow2, odataDevice, idataDevice, iblockSumDevice);

            // Exclusive scan of the final prefix sums computed by each block. oblockSumDevice
            // will store the increments to be added to each block's scan results.
            exclusiveScanKernel<<<blockSumBlockCount, blockSumBlockSize, blockSumSharedMemSize>>>(blockCount.x, oblockSumDevice, iblockSumDevice);

            // Add the block increments to the original scan results to obtain final results.
            addBlockIncrementsKernel<<<blockCount, blockSize>>>(nNextPow2, odataDevice, oblockSumDevice);

            cudaMemcpy(odata, odataDevice, n * sizeof(int), cudaMemcpyDeviceToHost);

            cudaFree(oblockSumDevice);
            checkCUDAError("failed to free oblockSumDevice");
            cudaFree(iblockSumDevice);
            checkCUDAError("failed to free iblockSumDevice");
            cudaFree(odataDevice);
            checkCUDAError("failed to free odataDevice");
            cudaFree(idataDevice);
            checkCUDAError("failed to free idataDevice");
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            timer().startGpuTimer();

            scanNoTimer(n, odata, idata);

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

            int blockSize = 256;
            int blockCount = (n + blockSize - 1) / blockSize;

            int *idataDevice = nullptr;
            cudaMalloc(&idataDevice, n * sizeof(int));
            checkCUDAError("failed to malloc idataDevice");
            cudaMemcpy(idataDevice, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            checkCUDAError("failed to memcpy idataDevice");

            int *nonzeroMaskDevice = nullptr;
            cudaMalloc(&nonzeroMaskDevice, n * sizeof(int));
            checkCUDAError("failed to malloc nonzeroMaskDevice");

            int *nonzeroMaskPrefixSumDevice = nullptr;
            cudaMalloc(&nonzeroMaskPrefixSumDevice, n * sizeof(int));
            checkCUDAError("failed to malloc nonzeroMaskPrefixSumDevice");

            StreamCompaction::Common::kernMapToBoolean<<<blockCount, blockSize>>>(n, nonzeroMaskDevice, idataDevice);

            scanNoTimer(n, nonzeroMaskPrefixSumDevice, nonzeroMaskDevice);

            int lastScanValue = 0;
            int lastMaskValue = 0;
            cudaMemcpy(&lastScanValue, &nonzeroMaskPrefixSumDevice[n - 1], sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(&lastMaskValue, &nonzeroMaskDevice[n - 1], sizeof(int), cudaMemcpyDeviceToHost);
            int nonzeroCount = lastScanValue + lastMaskValue;

            if (nonzeroCount > 0) {
                int *odataDevice = nullptr;
                cudaMalloc(&odataDevice, nonzeroCount * sizeof(int));
                checkCUDAError("failed to malloc odataDevice");

                StreamCompaction::Common::kernScatter<<<blockCount, blockSize>>>(
                    n, odataDevice, idataDevice, nonzeroMaskDevice, nonzeroMaskPrefixSumDevice
                );

                cudaMemcpy(odata, odataDevice, nonzeroCount * sizeof(int), cudaMemcpyDeviceToHost);
                checkCUDAError("failed to memcpy odataDevice");

                cudaFree(odataDevice);
                checkCUDAError("failed to free odataDevice");
            }

            cudaFree(nonzeroMaskPrefixSumDevice);
            checkCUDAError("failed to free nonzeroMaskPrefixSumDevice");
            cudaFree(nonzeroMaskDevice);
            checkCUDAError("failed to free nonzeroMaskDevice");
            cudaFree(idataDevice);
            checkCUDAError("failed to free idataDevice");

            timer().endGpuTimer();
            return nonzeroCount;
        }

        // Parallel reduction corresponding to up-sweep phase of scan. Standalone version
        // to test it.
        __global__ void reductionKernel(int n, int *odata, int *idata) {
            extern __shared__ int blockSharedMem[];

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

            odata[globalThreadIdx] = blockSharedMem[threadIdx.x];
        }

        // CPU version of the reduction kernel (up-sweep), for testing and comparison.
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
    }
}

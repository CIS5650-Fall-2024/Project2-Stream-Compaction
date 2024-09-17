#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"

namespace StreamCompaction {
    namespace Naive {
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
        
        // Performs exclusive scan on idata *per block*, not on the whole n elements at a time.
        // Uses shared memory to perform the scan at the block level, alternating 2 buffers for
        // reads and writes each iteration, before copying the resulting partial sums to odata
        // at the end.
        //
        // If blockSumDevice is not nullptr, the final prefix sum computed by the block is stored
        // in blockSumDevice[blockIdx.x].
        __global__ void exclusiveScanKernel(int n, int *odata, int *idata, int *blockSumDevice = nullptr) {
            // Shared memory buffer should be twice the size of the block, so that we may have 2
            // buffers to alternate reads and writes.
            extern __shared__ int blockSharedMem[];

            int globalThreadIdx = blockIdx.x*blockDim.x + threadIdx.x;
            if (globalThreadIdx >= n) {
                // Extra threads in the last block.
                return;
            }

            // For each depth d, iterInput is read from and iterOutput is written to
            // and then swapped.
            int *iterInput = blockSharedMem;
            int *iterOutput = blockSharedMem + blockDim.x;

            if (threadIdx.x == 0) {
                // Put addition identity in first element.
                iterInput[threadIdx.x] = 0;
            } else {
                // Copy input data to iterInput, shifting by 1. This effectively turns
                // an inclusive scan into an exclusive scan.
                iterInput[threadIdx.x] = idata[globalThreadIdx-1];
            }

            __syncthreads();

            int k = threadIdx.x;
            for (int d = 1; d <= ilog2ceil(n); ++d) {
                // Elements to be added are this much apart.
                int delta = 1 << (d-1);

                // At the beginning of each new iteration:
                //  - partial sums [0, 2^(d-1) - 1] are complete;
                //  - the rest are of the form x[k - 2^d - 1] + ... + x[k].

                if (k > delta) {
                    // Note that if k = delta, then iterInput[k - delta] = 0, so that's handled
                    // by the other case.
                    iterOutput[k] = iterInput[k - delta] + iterInput[k];
                } else {
                    iterOutput[k] = iterInput[k];
                }

                __syncthreads();

                int *tmp = iterInput;
                iterInput = iterOutput;
                iterOutput = tmp;
            }

            // iterInput now contains the final result of the scan for this block.
            //
            // The synchronization barrier at the end of the loop ensures that all threads
            // have finished writing to iterInput before we copy the results to odata; all
            // threads execute exactly the same number of iterations, so we don't need to
            // worry about threads in the same block being at different stages of the scan.
            odata[globalThreadIdx] = iterInput[threadIdx.x];

            // blockSumDevice will store the final prefix sums computed by all blocks to 
            // later combine all blocks' results.
            if (blockSumDevice != nullptr && threadIdx.x == blockDim.x - 1) {
                // An exclusive scan doesn't include the last element, so we need to add it.
                blockSumDevice[blockIdx.x] = iterInput[blockDim.x - 1] + idata[globalThreadIdx];
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

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            timer().startGpuTimer();

            // Kernel configuration for the block-level scan and block sum computation.
            const int blockSize = 256;
            dim3 blockCount = (n + blockSize - 1) / blockSize;
            // For double buffering, shared memory must be able to host 2 block-sized buffers.
            const int blockSharedMemSize = 2 * blockSize * sizeof(int);

            // Kernel configuration for the block sum scan (all blocks' final prefix sums).
            const int blockSumBlockSize = blockCount.x;
            dim3 blockSumBlockCount = (blockCount.x + blockSumBlockSize - 1) / blockSumBlockSize;
            // For double buffering, shared memory must be able to host 2 block-sized buffers.
            const int blockSumSharedMemSize = 2 * blockSumBlockSize * sizeof(int);

            int *idataDevice = nullptr;
            cudaMalloc(&idataDevice, n * sizeof(int));
            checkCUDAError("failed to malloc idataDevice");

            int *odataDevice = nullptr;
            cudaMalloc(&odataDevice, n * sizeof(int));
            checkCUDAError("failed to malloc odataDevice");

            int *iblockSumDevice = nullptr;
            cudaMalloc(&iblockSumDevice, blockCount.x * sizeof(int));
            checkCUDAError("failed to malloc iblockSumDevice");

            int *oblockSumDevice = nullptr;
            cudaMalloc(&oblockSumDevice, blockCount.x * sizeof(int));
            checkCUDAError("failed to malloc oblockSumDevice");

            cudaMemcpy(idataDevice, idata, n * sizeof(int), cudaMemcpyHostToDevice);

            // No need to synchronize here: the CPU blocks until the transfer is complete.

            // Per-block exclusive scan of the original input. iblockSumDevice will store the
            // final prefix sums computed by each block.
            exclusiveScanKernel<<<blockCount, blockSize, blockSharedMemSize>>>(n, odataDevice, idataDevice, iblockSumDevice);
            cudaDeviceSynchronize();

            // Exclusive scan of the final prefix sums computed by each block. oblockSumDevice
            // will store the increments to be added to each block's scan results.
            exclusiveScanKernel<<<blockSumBlockCount, blockSumBlockSize, blockSumSharedMemSize>>>(blockCount.x, oblockSumDevice, iblockSumDevice);
            cudaDeviceSynchronize();

            // Add the block increments to the original scan results to obtain final results.
            addBlockIncrementsKernel<<<blockCount, blockSize>>>(n, odataDevice, oblockSumDevice);
            cudaDeviceSynchronize();

            cudaMemcpy(odata, odataDevice, n * sizeof(int), cudaMemcpyDeviceToHost);

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
    }
}

#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"

#define blockSize 256

namespace StreamCompaction {
    namespace Naive {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }
        
        __global__ void runIter(int n, int d, int *odata, const int *idata) {
            int index = threadIdx.x + (blockIdx.x * blockDim.x);
            if (index >= n) {
                return;
            }
            int pow2d1 = 1 << (d - 1);
            if (index >= pow2d1) {
                odata[index] = idata[index - pow2d1] + idata[index];
            } else {
                odata[index] = idata[index];
            }
        }

        __global__ void convertInclusiveToExclusive(int n, int *odata, const int *idata) {
            int index = threadIdx.x + (blockIdx.x * blockDim.x);
            if (index >= n) {
                return;
            }
            odata[index] = index == 0 ? 0 : idata[index - 1];
        }

        __global__ void copyArray(int n, int *odata, const int *idata) {
            int index = threadIdx.x + (blockIdx.x * blockDim.x);
            if (index >= n) {
                return;
            }
            odata[index] = idata[index];
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            int *dev_idata;
            int *dev_odata;
            cudaMalloc((void**)&dev_idata, n * sizeof(int));
            cudaMalloc((void**)&dev_odata, n * sizeof(int));
            checkCUDAError("cudaMalloc failed");
            cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            checkCUDAError("cudaMemcpy failed");
            int numBlocks = (n + blockSize - 1) / blockSize;

            timer().startGpuTimer();
            int numIters = ilog2ceil(n);
            for (int d = 1; d <= numIters; d++) {
                runIter<<<numBlocks, blockSize>>>(n, d, dev_odata, dev_idata);
                std::swap(dev_odata, dev_idata);
            }
            convertInclusiveToExclusive<<<numBlocks, blockSize>>>(n, dev_odata, dev_idata);
            timer().endGpuTimer();

            cudaMemcpy(odata, dev_odata, n * sizeof(int), cudaMemcpyDeviceToHost);
            cudaFree(dev_idata);
            cudaFree(dev_odata);
            checkCUDAError("cudaFree failed");
        }

        __global__ void runScanSharedChunk(int n, int *odata, int *blockSums, const int *idata) {
            extern __shared__ int shared_data[];

            int index = threadIdx.x;
            int chunkSize = blockDim.x;
            int globalIndex = threadIdx.x + (blockIdx.x * blockDim.x);
            if (globalIndex >= n) {
                return;
            }

            shared_data[index] = idata[globalIndex];
            __syncthreads();

            for (int d = 1; d < chunkSize; d <<= 1) {
                int temp;
                if (index >= d) {
                    temp = shared_data[index - d] + shared_data[index];
                } else {
                    temp = shared_data[index];
                }
                __syncthreads();
                shared_data[index] = temp;
                __syncthreads();
            }

            odata[globalIndex] = shared_data[index];
            if (blockSums != nullptr && index == chunkSize - 1) {
                blockSums[blockIdx.x] = shared_data[index];
            }
        }

        __global__ void scanSharedAddBlockSums(int n, int *data, const int *blockSums) {
            int index = threadIdx.x + (blockIdx.x * blockDim.x);
            if (index >= n) {
                return;
            }
            data[index] += blockSums[blockIdx.x];
        }

        void scanSharedHelper(int n, int *dev_odata, int *dev_idata) {
            int numBlocks = (n + blockSize - 1) / blockSize;

            if (numBlocks > 1) {
                int *dev_blockSums;
                int *dev_blockSumsScan;
                cudaMalloc((void**)&dev_blockSums, numBlocks * sizeof(int));
                cudaMalloc((void**)&dev_blockSumsScan, numBlocks * sizeof(int));

                runScanSharedChunk<<<numBlocks, blockSize, blockSize * sizeof(int)>>>(n, dev_odata, dev_blockSums, dev_idata);
                scanSharedHelper(numBlocks, dev_blockSumsScan, dev_blockSums);
                scanSharedAddBlockSums<<<numBlocks, blockSize>>>(n, dev_odata, dev_blockSumsScan);

                cudaFree(dev_blockSums);
                cudaFree(dev_blockSumsScan);
            } else {
                int numThreads = std::min(n, blockSize);
                runScanSharedChunk<<<1, numThreads, numThreads * sizeof(int)>>>(n, dev_odata, nullptr, dev_idata);
            }
            copyArray<<<numBlocks, blockSize>>>(n, dev_idata, dev_odata);
            convertInclusiveToExclusive<<<numBlocks, blockSize>>>(n, dev_odata, dev_idata);
        }

        void scanShared(int n, int *odata, const int *idata) {
            int *dev_idata;
            int *dev_odata;
            cudaMalloc((void**)&dev_idata, n * sizeof(int));
            cudaMalloc((void**)&dev_odata, n * sizeof(int));
            checkCUDAError("cudaMalloc failed");
            cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            checkCUDAError("cudaMemcpy failed");

            timer().startGpuTimer();
            scanSharedHelper(n, dev_odata, dev_idata);
            timer().endGpuTimer();

            cudaMemcpy(odata, dev_odata, n * sizeof(int), cudaMemcpyDeviceToHost);
            cudaFree(dev_idata);
            cudaFree(dev_odata);
            checkCUDAError("cudaFree failed");
        }
    }
}

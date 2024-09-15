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
        
        __global__ void exclusiveScanKernel(int n, int *odata, int *idata, int *blockSumDevice = nullptr) {
            extern __shared__ int blockSharedMem[];

            int globalThreadIdx = blockIdx.x*blockDim.x + threadIdx.x;
            if (globalThreadIdx >= n) {
                return;
            }

            int *iterInput = blockSharedMem;
            int *iterOutput = blockSharedMem + blockDim.x;

            if (threadIdx.x == 0) {
                iterInput[threadIdx.x] = 0;
            } else {
                iterInput[threadIdx.x] = idata[globalThreadIdx-1];
            }

            __syncthreads();

            int k = threadIdx.x;
            for (int d = 1; d <= ilog2ceil(n); ++d) {
                if (k >= pow(2, d-1)) {
                    iterOutput[k] = iterInput[k - (int)pow(2, d-1)] + iterInput[k];
                } else {
                    iterOutput[k] = iterInput[k];
                }

                __syncthreads();

                int *tmp = iterInput;
                iterInput = iterOutput;
                iterOutput = tmp;
            }

            odata[globalThreadIdx] = iterInput[threadIdx.x];

            if (blockSumDevice != nullptr && threadIdx.x == 0) {
                blockSumDevice[blockIdx.x] = iterInput[blockDim.x - 1];
            }
        }

        __global__ void addBlockSumsKernel(int n, int *odata, int *blockSums) {
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

            const int blockSize = 256;
            dim3 blockCount = (n + blockSize - 1) / blockSize;
            const int blockSharedMemSize = 2 * blockSize * sizeof(int);

            const int blockSumBlockSize = blockCount.x;
            dim3 blockSumBlockCount = (blockCount.x + blockSumBlockSize - 1) / blockSumBlockSize;
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

            // Debug.
            int *odataDeviceDebug = (int *)malloc(n * sizeof(int));
            int *iblockSumDeviceDebug = (int *)malloc(blockCount.x * sizeof(int));
            int* oblockSumDeviceDebug = (int *)malloc(blockCount.x * sizeof(int));
            int* odataDeviceAfterSumDebug = (int*)malloc(n * sizeof(int));
            // Debug.

            cudaMemcpy(idataDevice, idata, n * sizeof(int), ::cudaMemcpyHostToDevice);

            exclusiveScanKernel<<<blockCount, blockSize, blockSharedMemSize>>>(n, odataDevice, idataDevice, iblockSumDevice);

            // Debug.
            cudaMemcpy(odataDeviceDebug, odataDevice, n * sizeof(int), ::cudaMemcpyDeviceToHost);
            cudaMemcpy(iblockSumDeviceDebug, iblockSumDevice, blockCount.x * sizeof(int), ::cudaMemcpyDeviceToHost);
            // Debug.

            exclusiveScanKernel<<<blockSumBlockCount, blockSumBlockSize, blockSumSharedMemSize>>>(blockCount.x, oblockSumDevice, iblockSumDevice);

            // Debug.
            cudaMemcpy(oblockSumDeviceDebug, oblockSumDevice, blockCount.x * sizeof(int), ::cudaMemcpyDeviceToHost);
            // Debug.

            addBlockSumsKernel<<<blockCount, blockSize>>>(n, odataDevice, oblockSumDevice);

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
    }
}

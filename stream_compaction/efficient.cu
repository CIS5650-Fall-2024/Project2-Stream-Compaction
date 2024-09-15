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

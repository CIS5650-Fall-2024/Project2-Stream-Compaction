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
        
        __global__ void exclusiveScanKernel(int n, int *odata, int *idata) {
            int globalThreadIdx = blockIdx.x*blockDim.x + threadIdx.x;
            if (globalThreadIdx >= n) {
                return;
            }

            int *iterInput = odata;
            int *iterOutput = idata;

            if (globalThreadIdx == 0) {
                iterInput[0] = 0;
            } else {
                iterInput[globalThreadIdx] = idata[globalThreadIdx-1];
            }

            __syncthreads();

            int k = globalThreadIdx;
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

            if (iterInput != odata) {
                odata[globalThreadIdx] = iterInput[globalThreadIdx];
            }
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            timer().startGpuTimer();

            int *idataDevice = nullptr;
            cudaMalloc(&idataDevice, n * sizeof(idata));
            checkCUDAError("failed to malloc idataDevice");

            int *odataDevice = nullptr;
            cudaMalloc(&odataDevice, n * sizeof(idata));
            checkCUDAError("failed to malloc odataDevice");

            cudaMemcpy(idataDevice, idata, n * sizeof(int), ::cudaMemcpyHostToDevice);

            const int blockSize = 256;
            dim3 blockCount = (n + blockSize - 1) / blockSize;
            exclusiveScanKernel<<<blockCount, blockSize>>>(n, odataDevice, idataDevice);

            cudaMemcpy(odata, odataDevice, n * sizeof(int), ::cudaMemcpyDeviceToHost);

            cudaFree(odataDevice);
            checkCUDAError("failed to free odataDevice");

            cudaFree(idataDevice);
            checkCUDAError("failed to free idataDevice");

            timer().endGpuTimer();
        }
    }
}

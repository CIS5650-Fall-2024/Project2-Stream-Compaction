#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"
#include <iostream>

namespace StreamCompaction {
    namespace Naive {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }
        // TODO: __global__
        __global__ void parallelScan(int n, int* odata, const int* idata, int level) {
            int index = threadIdx.x + (blockIdx.x * blockDim.x);
            if (index >= n) {
                return;
            }
            int levelOffset = 1 << (level - 1);
            int valueToAdd = (index >= levelOffset) ? idata[index - levelOffset] : 0;
            odata[index] = valueToAdd + idata[index];
        }

        __global__ void inclusiveToExclusive(int n, int* odata, const int* idata) {
            int index = threadIdx.x + (blockIdx.x * blockDim.x);
            if (index >= n) {
                return;
            }
            int data = (index == 0) ? 0 : idata[index - 1];
            odata[index] = data;
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            int* dev_idata;
            int* dev_odata;
            cudaMalloc((void**)&dev_idata, n * sizeof(int));
            checkCUDAError("cudaMalloc dev_idata failed!");
            cudaMalloc((void**)&dev_odata, n * sizeof(int));
            checkCUDAError("cudaMalloc dev_odata failed!");
            cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);

            dim3 fullBlocksPerGrid((n + blockSize - 1) / blockSize);
            int numLevels = ilog2ceil(n);

            timer().startGpuTimer();
            for (int i = 1; i <= numLevels; ++i) {
                parallelScan <<<fullBlocksPerGrid, blockSize>>> (n, dev_odata, dev_idata, i);
                std::swap(dev_idata, dev_odata);
            }
            inclusiveToExclusive <<<fullBlocksPerGrid, blockSize>>> (n, dev_odata, dev_idata);
            timer().endGpuTimer();

            cudaDeviceSynchronize();
            cudaMemcpy(odata, dev_odata, n * sizeof(int), cudaMemcpyDeviceToHost);
            cudaFree(dev_idata);
            cudaFree(dev_odata);
        }
    }
}

#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"

#define blockSize 128

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
            if (index == 0) {
                odata[index] = 0;
            } else {
                odata[index] = idata[index - 1];
            }
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
    }
}

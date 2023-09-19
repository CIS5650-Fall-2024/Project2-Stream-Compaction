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

        __global__ void kernScan(int n, int *odata, const int *idata, int offset) {
            int index = (blockIdx.x * blockDim.x) + threadIdx.x;
            if (index < n) {
                odata[index] = idata[index] + (index >= offset ? idata[index - offset] : 0);
            }
        }

        __global__ void kernShift(int n, int *odata, const int *idata) {
            int index = (blockIdx.x * blockDim.x) + threadIdx.x;
            if (index < n) {
                odata[index] = index ? idata[index - 1] : 0;
            }
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            int *dev_odata, *dev_idata;
            int rounds = ilog2ceil(n);
            cudaMalloc((void**)&dev_odata, n * sizeof(int));
            checkCUDAError("cudaMalloc dev_odata failed!");
            cudaMalloc((void**)&dev_idata, n * sizeof(int));
            checkCUDAError("cudaMalloc dev_idata failed!");
            cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            dim3 blocks((n + blockSize - 1) / blockSize);

            timer().startGpuTimer();
            int offset = 1;
            for (int i = 0; i < rounds; i++) {
                kernScan<<<blocks, blockSize>>>(n, dev_odata, dev_idata, offset);
                checkCUDAError("kernScan failed!");
                std::swap(dev_odata, dev_idata);
                offset <<= 1;
            }
            kernShift<<<blocks, blockSize>>>(n, dev_odata, dev_idata);
            checkCUDAError("kernShift failed!");
            timer().endGpuTimer();
            cudaMemcpy(odata, dev_odata, n * sizeof(int), cudaMemcpyDeviceToHost);
            cudaFree(dev_odata);
            cudaFree(dev_idata);
        }
    }
}

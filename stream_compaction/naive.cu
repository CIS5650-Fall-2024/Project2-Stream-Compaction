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

        __global__ void kernScanStep(const int n, const int offset, int* odata, const int* idata) {
            int k = (blockIdx.x * blockDim.x) + threadIdx.x;
            if (k >= n) {
                return;
            }

            if (k >= offset) {
                odata[k] = idata[k - offset] + idata[k];
            } else {
                odata[k] = idata[k];
            }
        }

        __global__ void kernInclusiveToExclusive(const int n, int* odata, const int* idata) {
            int k = (blockIdx.x * blockDim.x) + threadIdx.x;
            if (k >= n) {
                return;
            }

            if (k == 0) {
                odata[k] = 0;
            } else {
                odata[k] = idata[k - 1];
            }
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            int* dev_idata;
            int* dev_odata;

            cudaMalloc((void**)&dev_idata, n * sizeof(int));
            checkCUDAErrorFn("cudaMalloc dev_idata failed!");
            cudaMalloc((void**)&dev_odata, n * sizeof(int));
            checkCUDAErrorFn("cudaMalloc dev_odata failed!");

            cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            checkCUDAErrorFn("cudaMempcy to device failed!");

            dim3 fullBlocksPerGrid((n + blockSize - 1) / blockSize);

            timer().startGpuTimer();

            for (int d = 1; d <= ilog2ceil(n); ++d) {
                kernScanStep<<<fullBlocksPerGrid, blockSize>>>(n, 1 << (d - 1), dev_odata, dev_idata);
                std::swap(dev_idata, dev_odata);
            }

            kernInclusiveToExclusive<<<fullBlocksPerGrid, blockSize>>>(n, dev_odata, dev_idata);

            timer().endGpuTimer();

            cudaMemcpy(odata, dev_odata, n * sizeof(int), cudaMemcpyDeviceToHost);
            checkCUDAErrorFn("cudaMempcy to host failed!");

            cudaFree(dev_idata);
            cudaFree(dev_odata);
            checkCUDAErrorFn("cudaFree failed!");
        }
    }
}

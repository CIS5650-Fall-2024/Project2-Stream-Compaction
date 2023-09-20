#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"

#define logBlockSize 6

namespace StreamCompaction {
    namespace Naive {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        __global__ void kernAdd(int d, int* odata, const int* idata) {
            int k = blockIdx.x * blockDim.x + threadIdx.x;
            int space = 1 << (d - 1);
            if (k >= space) {
                odata[k] = idata[k] + idata[k - space];
            }
            else {
                odata[k] = idata[k];
            }
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int* odata, const int* idata) {
            const int blockSize = 1 << logBlockSize;

            int* dev_odata;
            int* dev_idata;

            cudaMalloc((void**)&dev_odata, n * sizeof(int));
            cudaMalloc((void**)&dev_idata, n * sizeof(int));

            cudaMemcpy(dev_odata, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);

            timer().startGpuTimer();

            dim3 fullBlocksPerGrid((n + blockSize - 1) / blockSize);
            for (int i = 1; i <= ilog2ceil(n); i++) {
                kernAdd << <fullBlocksPerGrid, blockSize >> > (i, dev_odata, dev_idata);
                std::swap(dev_odata, dev_idata);
            }

            timer().endGpuTimer();

            cudaMemcpy(odata + 1, dev_idata, (n - 1) * sizeof(int), cudaMemcpyDeviceToHost);

            cudaFree(dev_odata);
            cudaFree(dev_idata);
            
        }
    }
}

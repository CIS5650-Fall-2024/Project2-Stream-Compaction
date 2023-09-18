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
        // TODO: __global__
        __global__ void kernNaiveGPUScan(int n, int d, int* odata, int* idata) 
        {
            int k = (blockIdx.x * blockDim.x) + threadIdx.x;
            if (k >= n) return;

            int powerd = 1 << (d - 1);
            if (k >= powerd) 
            {
                odata[k] = idata[k - powerd] + idata[k];
            }
            else 
            {
                odata[k] = idata[k];
            }
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            int* dev_odata;
            int* dev_idata;

            cudaMalloc((void**)&dev_odata, n * sizeof(int));
            checkCUDAErrorFn("cudaMalloc dev_odata failed!");
            cudaMalloc((void**)&dev_idata, n * sizeof(int));
            checkCUDAErrorFn("cudaMalloc dev_idata failed!");

            cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            checkCUDAErrorFn("cudaMemcpy idata to dev_idata failed!");

            dim3 fullBlocksPerGrid((n + blockSize - 1) / blockSize);

            timer().startGpuTimer();
            // TODO
            for (int d = 1; d <= ilog2ceil(n); d++) 
            {
                kernNaiveGPUScan<<<fullBlocksPerGrid, blockSize>>>(n, d, dev_odata, dev_idata);
                std::swap(dev_odata, dev_idata);
            }
            timer().endGpuTimer();

            cudaMemcpy(odata + 1, dev_idata, (n - 1) * sizeof(int), cudaMemcpyDeviceToHost);
            checkCUDAErrorFn("cudaMemcpy dev_idata to odata failed!");
            odata[0] = 0;

            cudaFree(dev_odata);
            checkCUDAErrorFn("cudaFree dev_odata failed!");
            cudaFree(dev_idata);
            checkCUDAErrorFn("cudaFree dev_idata failed!");
        }
    }
}

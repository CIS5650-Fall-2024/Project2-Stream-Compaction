#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"

#include <device_launch_parameters.h>

#define blockSize 256

namespace StreamCompaction {
    namespace Naive {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        /**
        * Kernel function for scan
        */
        __global__ void kernScan(int n, int d, int* odata, int* idata) {
            
            int k = (blockIdx.x * blockDim.x) + threadIdx.x;
            int off = 1 << (d - 1);

            if (k >= n) {
                return; 
            }
            else if (k >= off) {
                odata[k] = idata[k] + idata[k - off];
            }
            else {
                odata[k] = idata[k];
            }

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

            timer().startGpuTimer();
            
            // kernel invocations
            for (int d = 1; d <= ilog2ceil(n); d++) {
                kernScan<<<fullBlocksPerGrid, blockSize>>>(n, d, dev_odata, dev_idata);
                checkCUDAError("kernScan failed!");

                std::swap(dev_odata, dev_idata);
            }

            timer().endGpuTimer();

            cudaMemcpy(odata + 1, dev_idata, (n-1) * sizeof(int), cudaMemcpyDeviceToHost);

            cudaFree(dev_idata);
            cudaFree(dev_odata);
        }
    }
}

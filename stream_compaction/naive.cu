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
        // TODO: __global__

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            // TODO
            int* dev_odata1;
            cudaMalloc(&dev_odata1, n*sizeof(int));
            checkCUDAError("cudaMalloc dev_odata1 failed");
            cudaMemcpy(dev_odata1, idata, n*sizeof(int), cudaMemcpyKind::cudaMemcpyHostToDevice);
            checkCUDAError("cudaMemcpy from idata to dev_odata1 failed");
            int *dev_odata2;
            cudaMalloc(&dev_odata2, n*sizeof(int));
            checkCUDAError("cudaMalloc dev_odata2 failed");
            cudaMemcpy(dev_odata2, idata, n*sizeof(int), cudaMemcpyKind::cudaMemcpyHostToDevice);
            checkCUDAError("cudaMemcpy from idata to dev_odata2 failed");

            timer().startGpuTimer();
            dim3 fullBlocksPerGrid((n + blockSize - 1) / blockSize + 1);
            int d = ilog2ceil(n);
            int offset = 1;
            for (int k = 1; k <= d; ++k) {
                //offset <<= 1;
                kernScan << <fullBlocksPerGrid, blockSize >> > (n, (1 << (k-1)), dev_odata1, dev_odata2);
                std::swap(dev_odata1, dev_odata2);
            }
            checkCUDAError("kernScan failed");

            timer().endGpuTimer();
            cudaMemcpy(odata, dev_odata1, n*sizeof(int), cudaMemcpyKind::cudaMemcpyDeviceToHost);
            for (int i = n - 1; i >= 0; --i) {
                if (i == 0) odata[i] = 0;
                else odata[i] = odata[i - 1];
            }

            checkCUDAError("cudaMemcpy from dev_odata1 to odata failed");

            cudaFree(dev_odata1);
            cudaFree(dev_odata2);

        }

        __global__ void kernScan(int n, int offset , int *odata1, int * odata2 ) {
            int index = (blockDim.x * blockIdx.x) + threadIdx.x;
            if (index >= n) return;
            if (index >= offset) { /* no need to check if (index < n)*/
                //odata2[index + offset] = odata1[index] + odata1[index + offset];
                odata2[index] = odata1[index] + odata1[index - offset];
            }
            else {
                odata2[index] = odata1[index];
            }
        }
    }
}

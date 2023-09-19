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
        __global__ void kernNaiveScan(int *idata, int *odata, int n, int offset) {  
            int index = threadIdx.x + (blockIdx.x * blockDim.x);
            if (index >= n) return;
            if (index >= offset) {
                odata[index] = idata[index] + idata[index - offset];
            }
            else {
                odata[index] = idata[index];
            }   
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            // TODO
            int blockSize = 128;
            dim3 fullBlocksPerGrid((n + blockSize - 1) / blockSize);

            int* iarray;
            int* oarray;
            cudaMalloc((void**)&iarray, n * sizeof(int));
            cudaMalloc((void**)&oarray, n * sizeof(int));
            cudaMemcpy(iarray, idata, n * sizeof(int), cudaMemcpyHostToDevice);

            timer().startGpuTimer();
            for (int i = 1; i <= ilog2ceil(n); i++)   {
                int offset = pow(2, i - 1);
                kernNaiveScan <<<fullBlocksPerGrid, blockSize>>> (iarray, oarray, n, offset);
                cudaMemcpy(iarray, oarray, n * sizeof(int), cudaMemcpyDeviceToDevice);
            }         
            timer().endGpuTimer();

            cudaMemcpy(odata + 1, oarray, n * sizeof(int), cudaMemcpyDeviceToHost);
            odata[0] = 0;
            cudaFree(iarray);
            cudaFree(oarray);
        }
    }
}

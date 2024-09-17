#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"
#include "efficient.h"

namespace StreamCompaction {
    namespace Naive {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }
        __global__ void kernNaiveScan(int offset, int n, int* odata, int* idata) {
            int k = threadIdx.x + blockIdx.x * blockDim.x;
            if (k >= n) return;
            if (k >= offset) {
                int index = k - (offset);
                odata[k] = idata[index] + idata[k];
            }
            else {
                odata[k] = idata[k];
            }
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int* odata, const int* idata) {
            int* dev_odata;
            int* dev_idata;
            // Allocate memory to GPU
            cudaMalloc((void**)&dev_odata, n * sizeof(int));
            checkCUDAError("cudaMalloc dev_odata failed!");
            cudaMalloc((void**)&dev_idata, n * sizeof(int));
            checkCUDAError("cudaMalloc dev_idata failed!");

            cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            checkCUDAError("cudaMemcpy idata failed!");
            //blockSize = 256?
            // what block size to use?
            int blockSize = 256;
            dim3 fullBlocksPerGrid((n + blockSize - 1) / blockSize);

            timer().startGpuTimer();
            for (int d = 1; d <= ilog2ceil(n); ++d) {
                int offset = 1 << (d - 1);
                kernNaiveScan << <fullBlocksPerGrid, blockSize >> > (offset, n, dev_odata, dev_idata);
                checkCUDAError("kernNaiveScan failed!");
                // Swap dev_odata and dev_idata
                int* temp = dev_idata;
                dev_idata = dev_odata;
                dev_odata = temp;
            }
            timer().endGpuTimer();
            // Copy data from GPU to CPU
            // inclusive scan to exclusive scan
            odata[0] = 0;
            cudaMemcpy(odata + 1, dev_idata, n * sizeof(int), cudaMemcpyDeviceToHost);
            checkCUDAError("cudaMemcpy odata failed!");
            // Free memory
            cudaFree(dev_odata);
            cudaFree(dev_idata);
        }

    }
}

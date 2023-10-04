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

        // Performs one iteration of naive scan of N elements.
        __global__ void kernNaiveScan(int n, int offset, int *odata, const int *idata) {
            int index = blockIdx.x * blockDim.x + threadIdx.x;
            if (index >= n) {
                return;
            }

            if (index >= offset) {
                odata[index] = idata[index - offset] + idata[index];
            }
            else {
                odata[index] = idata[index];
            }
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            dim3 blocksPerGrid((n + blockSize - 1) / blockSize);
            
            // copy data to the GPU
            int* dev_buf1;
            int* dev_buf2;

            size_t arrSize = n * sizeof(int);
            cudaMalloc((void**)&dev_buf1, arrSize);
            checkCUDAError("cudaMalloc dev_buf1 failed!");
            cudaMalloc((void**)&dev_buf2, arrSize);
            checkCUDAError("cudaMalloc dev_buf2 failed!");

            cudaMemcpy(dev_buf1, idata, arrSize, cudaMemcpyHostToDevice);
            checkCUDAError("cudaMemcpy idata to dev_buf1 failed!");

            timer().startGpuTimer();
            for (int d = 1; d <= ilog2ceil(n); ++d) {
                int offset = 1 << (d - 1);
                kernNaiveScan <<<blocksPerGrid, blockSize>>> (n, offset, dev_buf2, dev_buf1);
                checkCUDAError("kernNaiveScan failed!");

                // ping-pong buffer to avoid race conditions
                int* tmp = dev_buf1;
                dev_buf1 = dev_buf2;
                dev_buf2 = tmp;
            }
            timer().endGpuTimer();

            // shift inclusive to exclusive scan for compaction
            odata[0] = 0;
            cudaMemcpy(odata + 1, dev_buf1, arrSize, cudaMemcpyDeviceToHost);
            checkCUDAError("cudaMemcpy odata failed!");

            // free memory
            cudaFree(dev_buf2);
            cudaFree(dev_buf1);
        }
    }
}

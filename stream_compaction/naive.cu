#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"
#include "device_launch_parameters.h"

namespace StreamCompaction {
    namespace Naive {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        // TODO: __global__
        __global__ void kernelNaiveScanAdd(int N, int step, int* odata, const int* idata) {
            int index = threadIdx.x + (blockIdx.x * blockDim.x);
            if (index < N) {
                if (index >= step) odata[index] = idata[index] + idata[index - step];
                else odata[index] = idata[index];
            }
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            //device memory initialized
            int* dev_tmp1;
            int* dev_tmp2;
            size_t arrSize = n * sizeof(int);
            cudaMalloc((void**)&dev_tmp1, arrSize);
            checkCUDAError("cudaMalloc dev_tmp1 failed!");
            cudaMalloc((void**)&dev_tmp2, arrSize);
            checkCUDAError("cudaMalloc dev_tmp2 failed!");
            // copy array from cpu to gpu
            cudaMemcpy(dev_tmp1, idata, arrSize, cudaMemcpyHostToDevice);
            checkCUDAError("cudaMemcpy idata to dev_tmp1 failed!");

            // iteration initialization
            int iterNum = ilog2ceil(n);
            int blockNum((n + blockSize - 1) / blockSize);
            
            timer().startGpuTimer();
            for (int d = 1; d <= iterNum; d++) {
                int step = 1 << (d - 1);
                kernelNaiveScanAdd<<<blockNum, blockSize>>>(n, step, dev_tmp2, dev_tmp1);
                checkCUDAError("kernelNaiveScanAdd failed!");
                std::swap(dev_tmp1, dev_tmp2);
            }

            timer().endGpuTimer();

            //inclusive to exclusive, copy array from gpu to cpu
            odata[0] = 0;
            cudaMemcpy(odata + 1, dev_tmp1, arrSize, cudaMemcpyDeviceToHost);
            checkCUDAError("cudaMemcpy final odata failed!");

            // free memory
            cudaFree(dev_tmp1);
            cudaFree(dev_tmp2);
        }
    }
}

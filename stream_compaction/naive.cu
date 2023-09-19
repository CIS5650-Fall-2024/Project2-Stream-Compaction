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

        __global__ void kernNaiveScan(int n, int d, int* odata, int* idata) {
            int k = (blockIdx.x * blockDim.x) + threadIdx.x;
            if (k >= 1 << (d-1)) {
                odata[k] = idata[k-(1<<(d-1))] + idata[k];
            }
            else {
                odata[k] = idata[k];
            }
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            timer().startGpuTimer();
            int block_size = 128;
            int n_padded = 1 << ilog2ceil(n);
            int num_blocks = (n_padded + block_size - 1) / block_size;
            int* dev_bufferA;
            int* dev_bufferB;

            cudaMalloc((void**)&dev_bufferA, n_padded * sizeof(int));
            checkCUDAError("cudaMalloc dev_bufferA failed!");
            cudaMalloc((void**)&dev_bufferB, n_padded * sizeof(int));
            checkCUDAError("cudaMalloc dev_bufferB failed!");

            cudaMemset(dev_bufferA, 0, n_padded * sizeof(int));
            checkCUDAError("cudaMemset dev_bufferA failed!");
            cudaMemset(dev_bufferB, 0, n_padded * sizeof(int));
            checkCUDAError("cudaMemset dev_bufferB failed!");
            
            cudaMemcpy(dev_bufferA, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            checkCUDAError("cudaMemcpy idata to dev_bufferA failed!");
            
            int d_max = ilog2ceil(n_padded);
            for (int d=1; d<=d_max; d++) {
                kernNaiveScan<<<num_blocks, block_size>>>(n_padded, d, dev_bufferB, dev_bufferA);
                checkCUDAError("kernNaiveScan failed!");
                std::swap(dev_bufferA, dev_bufferB);
            }
            if (d_max % 2 == 0) {
                cudaMemcpy(odata, dev_bufferA, n * sizeof(int), cudaMemcpyDeviceToHost);
                checkCUDAError("cudaMemcpy dev_bufferA to odata failed!");
            } else {
                cudaMemcpy(odata, dev_bufferB, n * sizeof(int), cudaMemcpyDeviceToHost);
                checkCUDAError("cudaMemcpy dev_bufferB to odata failed!");
            }
            
            cudaFree(dev_bufferA);
            checkCUDAError("cudaFree dev_bufferA failed!");
            cudaFree(dev_bufferB);
            checkCUDAError("cudaFree dev_bufferB failed!");

            timer().endGpuTimer();
        }
    }
}

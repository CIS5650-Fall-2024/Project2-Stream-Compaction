#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"

#define block_size 128

namespace StreamCompaction {
    namespace Naive {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }
        __global__ void kernNaiveScanIter(int n, int* odata, int* idata, int d) {
            int index = threadIdx.x + (blockIdx.x * blockDim.x);
            if (index >= n) {
                return;
            }
            int offset = 1 << (d - 1);
            odata[index] = index < offset ? idata[index] : (idata[index - offset] + idata[index]);
        }

        __global__ void kernIncToExc(int n, int* odata, int* idata) {
            int index = threadIdx.x + (blockIdx.x * blockDim.x);
            if (index >= n) {
                return;
            }
            odata[index] = index == 0 ? 0 : idata[index - 1];
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            dim3 block_dim((n + block_size - 1) / block_size);
            int* dev_idata;
            int* dev_odata;
            cudaMalloc((void**)&dev_idata, n * sizeof(int));
            checkCUDAError("cudaMalloc dev_idata failed!");
            cudaMalloc((void**)&dev_odata, n * sizeof(int));
            checkCUDAError("cudaMalloc dev_odata failed!");
            cudaMemcpy(dev_idata, idata, sizeof(int) * n, cudaMemcpyHostToDevice);
            checkCUDAError("cudaMemcpy idata into dev_idata failed!");

            timer().startGpuTimer();
            for (int i = 1; i <= ilog2ceil(n); i++) {
                kernNaiveScanIter << <block_dim, block_size >> > (n, dev_odata, dev_idata, i);
                std::swap(dev_odata, dev_idata);
            }
            kernIncToExc << <block_dim, block_size >> > (n, dev_odata, dev_idata);
            timer().endGpuTimer();

            cudaMemcpy(odata, dev_odata, sizeof(int) * n, cudaMemcpyDeviceToHost);
            checkCUDAError("cudaMemcpy dev_odata into odata failed!");
            cudaFree(dev_odata);
            checkCUDAError("free dev_odata failed!");
            cudaFree(dev_idata);
            checkCUDAError("free dev_idata failed!");
        }
    }
}

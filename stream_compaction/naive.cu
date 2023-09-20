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
        __global__ void kernelScan(int n, int* odata, const int* idata, int offset)
        {
            int index = (blockDim.x * blockIdx.x) + threadIdx.x;
            if (index >= n)
            {
                return;     // invalid index
            }

            if (index >= offset)
            {
                odata[index] = idata[index - offset] + idata[index];
            }
            else
            {
                odata[index] = idata[index];
            }
        }

        __global__ void kernelShiftRight(int n, int* odata, int* idata)
        {
            int index = (blockDim.x * blockIdx.x) + threadIdx.x;
            if (index >= n)
            {
                return;     // invalid index
            }

            if (index == 0)
            {
                odata[index] = 0;
            }
            else
            {
                odata[index] = idata[index - 1];
            }
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            int* dev_idata;
            int* dev_odata;
            cudaMalloc((void**)&dev_idata, sizeof(int) * n);
            checkCUDAError("cudaMalloc dev_idata failed");
            cudaMalloc((void**)&dev_odata, sizeof(int) * n);
            checkCUDAError("cudaMalloc dev_odata failed");
            cudaMemcpy(dev_idata, idata, sizeof(int) * n, cudaMemcpyHostToDevice);
            checkCUDAError("cudaMemcpy dev_idata failed");
            cudaMemcpy(dev_odata, odata, sizeof(int) * n, cudaMemcpyHostToDevice);
            checkCUDAError("cudaMemcpy dev_odata failed");

            int totalBlocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
            timer().startGpuTimer();
            int max = ilog2ceil(n);
            for (int d = 1; d <= max; d++)
            {
                kernelScan<<<totalBlocks, BLOCK_SIZE>>>(n, dev_odata, dev_idata, 1 << d - 1);
                std::swap(dev_odata, dev_idata);
            }
            kernelShiftRight<<<totalBlocks, BLOCK_SIZE>>>(n, dev_odata, dev_idata);
            timer().endGpuTimer();
            cudaMemcpy(odata, dev_odata, sizeof(int) * n, cudaMemcpyDeviceToHost);
            checkCUDAError("cudaMemcpy dev_odata to odata failed");
            cudaFree(dev_idata);
            checkCUDAError("cudaFree dev_idata failed");
            cudaFree(dev_odata);
            checkCUDAError("cudaFree dev_odata failed");
        }
    }
}

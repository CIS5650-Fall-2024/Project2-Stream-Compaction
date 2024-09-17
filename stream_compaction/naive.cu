#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
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

        __global__ void kernScan(int n, int d, int* odata, int* tempdata)
        {
            unsigned int index = threadIdx.x + (blockIdx.x * blockDim.x);

            if (index >= n) return;
            if (index >= (unsigned int)(1 << d))
            {
                odata[index] = tempdata[index - (unsigned int)(1 << d)] + tempdata[index];
            }
            else
            {
                odata[index] = tempdata[index];
            }
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            int* dev_odata;
            int* dev_tempdata;

            cudaMalloc((void**)&dev_odata, sizeof(int) * n);
            checkCUDAError("cudaMalloc dev_odata failed!");
            cudaMalloc((void**)&dev_tempdata, sizeof(int) * n);
            checkCUDAError("cudaMlloc dev_tempdata failed!");
            cudaDeviceSynchronize();

            cudaMemcpy(dev_tempdata, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            cudaMemcpy(dev_odata, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            cudaDeviceSynchronize();

            unsigned int ilogn = ilog2ceil(n);
            dim3 blockNum((n + BLOCK_SIZE - 1) / BLOCK_SIZE);

            timer().startGpuTimer();
            // TODO
            for (int i = 0; i < ilogn; ++i)
            {
                kernScan << < blockNum, BLOCK_SIZE >> > (n, i, dev_odata, dev_tempdata);
                std::swap(dev_odata, dev_tempdata);
            }
            timer().endGpuTimer();

            std::swap(dev_odata, dev_tempdata);
            odata[0] = 0;
            cudaMemcpy(odata + 1, dev_odata, (n - 1) * sizeof(int), cudaMemcpyDeviceToHost);
            cudaFree(dev_odata);
            cudaFree(dev_tempdata);
        }
    }
}

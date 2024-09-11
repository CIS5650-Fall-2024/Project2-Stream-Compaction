#include <cuda.h>
#include "common.h"
#include "naive.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

constexpr int blockSize = 512;

namespace StreamCompaction {
    namespace Naive {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }
        // TODO: __global__

        __global__ void kernPartialSum(int N, int d, int* odata, const int *idata)
        {
            int index = (blockIdx.x * blockDim.x) + threadIdx.x;
            if (index >= N) { return; }
            if (index >= (1 << d))
            {
                odata[index] = idata[index - (1 << d)] + idata[index];
            }
            else
            {
                odata[index] = idata[index];
            }
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            timer().startGpuTimer();
            // TODO

            // Ping-pong device data buffers
            int* dev_idata;
            int* dev_odata;
            cudaMalloc((void**)&dev_idata, n * sizeof(int));
            cudaMalloc((void**)&dev_odata, n * sizeof(int));

            cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            cudaMemcpy(dev_odata, dev_idata, n * sizeof(int), cudaMemcpyDeviceToDevice);

            dim3 threadsPerBlock(blockSize);
            dim3 numBlocks((n + blockSize - 1) / blockSize);

            for (int d = 0; d < ilog2ceil(n); ++d)
            {
                kernPartialSum <<<numBlocks, threadsPerBlock>>> (n, d, dev_odata, dev_idata);
                checkCUDAError("Kernel launch failed.");
                std::swap(dev_odata, dev_idata);
            }

            odata[0] = 0;
            cudaMemcpy(odata + 1, dev_idata, (n - 1) * sizeof(int), cudaMemcpyDeviceToHost);

            cudaFree(dev_idata);
            cudaFree(dev_odata);
            timer().endGpuTimer();
        }
    }
}

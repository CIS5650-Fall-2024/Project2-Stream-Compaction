#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"

namespace StreamCompaction
{
    namespace Naive
    {
        const size_t g_blockSize = 128;

        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer &timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        struct TimerGuard
        {
            TimerGuard()
            {
                timer().startGpuTimer();
            }
            ~TimerGuard()
            {
                timer().endGpuTimer();
            }
        };

        __global__ void kernScan(int n, int *dst, int *src, int exp2d)
        {
            int k = blockIdx.x * blockDim.x + threadIdx.x;
            if (k >= n)
            {
                return;
            }
            dst[k] = src[k] + ((k >= exp2d) ? src[k - exp2d] : 0);
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata)
        {
            size_t numBytes = n * sizeof(int);
            dim3 gridDim((n + g_blockSize - 1) / g_blockSize);
            int iterations = ilog2ceil(n);

            int *dev_src;
            cudaMalloc(&dev_src, numBytes);
            cudaMemcpy(dev_src, idata, numBytes, cudaMemcpyHostToDevice);

            int *dev_dst;
            cudaMalloc(&dev_dst, numBytes);

            {
                TimerGuard _;
                for (int d = 0; d < iterations; d++)
                {
                    int exp2d = 1 << d;
                    kernScan<<<gridDim, g_blockSize>>>(n, dev_dst, dev_src, exp2d);
                    int buf[256];
                    cudaMemcpy(buf, dev_dst, numBytes, cudaMemcpyDeviceToHost);
                    std::swap(dev_src, dev_dst);
                }
            }

            odata[0] = 0;
            cudaMemcpy(odata + 1, dev_src, numBytes - sizeof(int), cudaMemcpyDeviceToHost);

            cudaFree(dev_src);
            cudaFree(dev_dst);
        }
    }
}
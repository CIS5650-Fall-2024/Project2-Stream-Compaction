#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

namespace StreamCompaction {
    namespace Efficient {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        __global__ void kernelUpSweep(int n, int* idata, int d)
        {
            int index = (blockDim.x * blockIdx.x) + threadIdx.x;
            if (index >= n)
            {
                return; // invalid index
            }
            
            int _d = 1 << d;            // 2^d
            int _d1 = 1 << (d + 1);      // 2^(d+1)
            if (index % _d1 == 0)    // TODO: avoid this? do this on the CPU?
            {
                idata[index + _d1 - 1] += idata[index + _d - 1];
            }
        }

        __global__ void kernelDownSweep(int n, int* idata, int d)
        {
            int index = (blockDim.x * blockIdx.x) + threadIdx.x;
            if (index >= n - 1)
            {
                return; // invalid index
            }

            int _d = 1 << d;            // 2^d
            int _d1 = 1 << (d + 1);      // 2^(d+1)
            if (index % _d1 == 0)
            {
                int left = idata[index + _d - 1];
                idata[index + _d - 1] = idata[index + _d1 - 1];
                idata[index + _d1 - 1] += left;
            }
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            int max = ilog2ceil(n);
            int nNextPowerOf2 = pow(2, max);

            int totalBlocks = (nNextPowerOf2 + BLOCK_SIZE - 1) / BLOCK_SIZE;

            int* dev_idata;
            cudaMalloc((void**)&dev_idata, sizeof(int) * nNextPowerOf2);
            checkCUDAError("cudaMalloc dev_idata failed");
            cudaMemcpy(dev_idata, idata, sizeof(int) * n, cudaMemcpyHostToDevice);
            checkCUDAError("cudaMemcpy dev_idata failed");


            timer().startGpuTimer();
            // TODO
            // upsweep

            for (int d = 0; d < max; d++)
            {
                kernelUpSweep<<<totalBlocks, BLOCK_SIZE>>>(nNextPowerOf2, dev_idata, d);
            }

            cudaMemset(dev_idata + nNextPowerOf2 - 1, 0, sizeof(int));

            // downsweep
            for (int d = max - 1; d >= 0; d--)
            {
                kernelDownSweep<<<totalBlocks, BLOCK_SIZE>>>(nNextPowerOf2, dev_idata, d);
            }
            timer().endGpuTimer();

            cudaMemcpy(odata, dev_idata, sizeof(int) * n, cudaMemcpyDeviceToHost);
            checkCUDAError("cudaMemcpy dev_idata to odata failed");
            cudaFree(dev_idata);
            checkCUDAError("cudaFree dev_idata failed");
        }

        /**
         * Performs stream compaction on idata, storing the result into odata.
         * All zeroes are discarded.
         *
         * @param n      The number of elements in idata.
         * @param odata  The array into which to store elements.
         * @param idata  The array of elements to compact.
         * @returns      The number of elements remaining after compaction.
         */
        int compact(int n, int *odata, const int *idata) {
            timer().startGpuTimer();
            // TODO
            timer().endGpuTimer();
            return -1;
        }
    }
}

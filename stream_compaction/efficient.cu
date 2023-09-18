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

            // Pad end elements with 0
            // For me this was not required and the "unset" elements were automatically defaulting to 0
            // But some compilers may throw this off so I'll just manually do the padding
            cudaMemset(&dev_idata[n], 0, sizeof(int) * (nNextPowerOf2 - n));
            checkCUDAError("cudaMemset dev_idata failed");

            timer().startGpuTimer();
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
            int* dev_idata;
            cudaMalloc((void**)&dev_idata, sizeof(int) * n);
            checkCUDAError("cudaMalloc dev_idata failed");
            cudaMemcpy(dev_idata, idata, sizeof(int) * n, cudaMemcpyHostToDevice);
            checkCUDAError("cudaMemcpy dev_idata failed");

            int* dev_odata;
            cudaMalloc((void**)&dev_odata, sizeof(int) * n);
            checkCUDAError("cudaMalloc dev_idata failed");

            int* dev_bools;
            cudaMalloc((void**)&dev_bools, sizeof(int) * n);
            checkCUDAError("cudaMalloc dev_bools failed");
            cudaMemset(dev_bools, -1, sizeof(int) * n);
            checkCUDAError("cudaMemset dev_bools failed");

            int totalBlocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

            int max = ilog2ceil(n);
            int nNextPowerOf2 = pow(2, max);
            int totalBlocksPow2 = (nNextPowerOf2 + BLOCK_SIZE - 1) / BLOCK_SIZE;


            int* dev_indices;
            cudaMalloc((void**)&dev_indices, sizeof(int) * nNextPowerOf2);
            checkCUDAError("cudaMalloc dev_indices failed");
            cudaMemset(dev_indices, -1, sizeof(int) * nNextPowerOf2);       // -1
            checkCUDAError("cudaMemset dev_indices failed");

            timer().startGpuTimer();
            // Map to bools array
            StreamCompaction::Common::kernMapToBoolean<<<totalBlocks, BLOCK_SIZE>>>(n, dev_bools, dev_idata);
            
            // Scan step
            // Copy bools to indices array to run scan on
            cudaMemcpy(dev_indices, dev_bools, sizeof(int) * n, cudaMemcpyDeviceToDevice);
            // Upsweep
            for (int d = 0; d < max; d++)
            {
                kernelUpSweep<<<totalBlocksPow2, BLOCK_SIZE>>> (nNextPowerOf2, dev_indices, d);
            }

            cudaMemset(dev_indices + nNextPowerOf2 - 1, 0, sizeof(int));
            checkCUDAError("cudaMemset dev_indices failed");

            // downsweep
            for (int d = max - 1; d >= 0; d--)
            {
                kernelDownSweep<<<totalBlocksPow2, BLOCK_SIZE>>>(nNextPowerOf2, dev_indices, d);
            }

            // Scatter step
            StreamCompaction::Common::kernScatter<<<totalBlocks, BLOCK_SIZE>>>(n, dev_odata, dev_idata, dev_bools, dev_indices);
            timer().endGpuTimer();

            int len;
            // I have no idea why I need this ternary check below
            cudaMemcpy(&len, dev_indices + n + (n == nNextPowerOf2 ? -1 : 0), sizeof(int), cudaMemcpyDeviceToHost);  // length of compacted array
            checkCUDAError("cudaMemcpy dev_indices last element failed");
            cudaMemcpy(odata, dev_odata, sizeof(int) * len, cudaMemcpyDeviceToHost);
            checkCUDAError("cudaMemcpy dev_odata failed");

            cudaFree(dev_idata);
            cudaFree(dev_odata);
            cudaFree(dev_bools);
            cudaFree(dev_indices);

            return len;
        }
    }
}

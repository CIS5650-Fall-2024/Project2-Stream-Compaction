#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

namespace StreamCompaction
{
    namespace Efficient
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

        __global__ void kernZero(int n, int *data)
        {
            int index = blockIdx.x * blockDim.x + threadIdx.x;
            if (index >= n)
            {
                return;
            }
            data[index] = 0;
        }

        __global__ void kernUpsweep(int n, int *data, int exp2d)
        {
            int index = blockIdx.x * blockDim.x + threadIdx.x;
            index = (index + 1) * exp2d * 2 - 1;
            if (index >= n)
            {
                return;
            }
            data[index] += data[index - exp2d];
        }

        __global__ void kernSetZeroSingle(int *data)
        {
            *data = 0;
        }

        __global__ void kernDownsweep(int n, int *data, int exp2d)
        {
            int index = blockIdx.x * blockDim.x + threadIdx.x;
            index = (index + 1) * exp2d * 2 - 1;
            if (index >= n)
            {
                return;
            }
            int tmp = data[index - exp2d];
            data[index - exp2d] = data[index];
            data[index] += tmp;
        }

        void dev_scan(int n, int *dev_data)
        {
            int iterations = ilog2ceil(n);
            int nCeil = 1 << iterations;
            for (int d = 0; d < iterations; d++)
            {
                int exp2d = 1 << d;
                int numThreads = nCeil / exp2d / 2;
                dim3 gridDim((numThreads + g_blockSize - 1) / g_blockSize);
                kernUpsweep<<<gridDim, g_blockSize>>>(nCeil, dev_data, exp2d);
            }
            kernSetZeroSingle<<<1, 1>>>(dev_data + nCeil - 1);
            for (int d = iterations - 1; d >= 0; d--)
            {
                int exp2d = 1 << d;
                int numThreads = nCeil / (exp2d * 2);
                dim3 gridDim((numThreads + g_blockSize - 1) / g_blockSize);
                kernDownsweep<<<gridDim, g_blockSize>>>(nCeil, dev_data, exp2d);
            }
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata)
        {
            int iterations = ilog2ceil(n);
            int nCeil = 1 << iterations;
            int *dev_data;

            size_t numBytes = n * sizeof(int);
            size_t numBytesCeil = nCeil * sizeof(int);

            cudaMalloc(&dev_data, numBytesCeil);
            cudaMemcpy(dev_data, idata, numBytes, cudaMemcpyHostToDevice);

            {
                TimerGuard _;
                dev_scan(n, dev_data);
            }

            cudaMemcpy(odata, dev_data, numBytes, cudaMemcpyDeviceToHost);
            cudaFree(dev_data);
        }

        __global__ void kernMapToBoolean(int n, int *src, int *dst) {
            int index = blockIdx.x * blockDim.x + threadIdx.x;
            if (index >= n) {
                return;
            }
            dst[index] = src[index] != 0 ? 1 : 0;
        }

        __global__ void kernScatter(int n, int *src, int *indices, int *dst) {
            int index = blockIdx.x * blockDim.x + threadIdx.x;
            if (src[index] == 0) {
                return;
            }
            dst[indices[index]] = src[index];
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
        int compact(int n, int *odata, const int *idata)
        {
            int iterations = ilog2ceil(n);
            int nCeil = 1 << iterations;
            
            size_t numBytes = n * sizeof(int);
            int *dev_idata;
            cudaMalloc(&dev_idata, numBytes);
            cudaMemcpy(dev_idata, idata, numBytes, cudaMemcpyHostToDevice);

            int *dev_odata;
            cudaMalloc(&dev_odata, numBytes);
            
            size_t numBytesCeil = nCeil * sizeof(int);
            int *dev_indices;
            cudaMalloc(&dev_indices, numBytesCeil);

            {
                TimerGuard _;
                dim3 gridDim((n + g_blockSize - 1) / g_blockSize);
                kernMapToBoolean<<<gridDim, g_blockSize>>>(n, dev_idata, dev_indices);
                dev_scan(n, dev_indices);
                kernScatter<<<gridDim, g_blockSize>>>(n, dev_idata, dev_indices, dev_odata);
            }

            int outSize;
            cudaMemcpy(&outSize, dev_indices + n - 1, sizeof(int), cudaMemcpyDeviceToHost);
            if (idata[n - 1] != 0) {
                outSize++;
            }
            cudaMemcpy(odata, dev_odata, outSize * sizeof(int), cudaMemcpyDeviceToHost);
            return outSize;
        }
    }
}

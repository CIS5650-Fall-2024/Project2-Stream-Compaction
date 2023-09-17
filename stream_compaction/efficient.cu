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

        __global__ void kernUpSweep(const int n, int* data, const int stride) {
            int index = blockIdx.x * blockDim.x + threadIdx.x + 1;
            if (index > n)return;
            int real_i = index * stride * 2 - 1;
            data[real_i] += data[real_i - stride];
        }

        __global__ void kernDownSweep(const int n, int* data, const int stride) {
            int index = blockIdx.x * blockDim.x + threadIdx.x + 1;
            if (index > n)return;
            int real_i = index * stride * 2 - 1;
            int t = data[real_i];
            data[real_i] += data[real_i - stride];
            data[real_i - stride] = t;
        }
        void scanInplace(int n, int* dev_data) {
            if (n != 1 << ilog2ceil(n))
                throw std::runtime_error("n is not pow of 2");
            int strideMax = n >> 1;
            dim3 fullBlocksPerGrid;
            for (int i = 1, int n = strideMax; i < strideMax; i <<= 1, n >>= 1)
            {
                fullBlocksPerGrid = ((n + blockSize - 1) / blockSize);
                kernUpSweep << <fullBlocksPerGrid, blockSize >> > (n, dev_data, i);
            }
            cudaMemset(&dev_data[n - 1], 0, sizeof(int));
            for (int i = strideMax, int n = 1; i >= 1; i >>= 1, n <<= 1)
            {
                fullBlocksPerGrid = ((n + blockSize - 1) / blockSize);
                kernDownSweep << <fullBlocksPerGrid, blockSize >> > (n, dev_data, i);
            }
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int* odata, const int* idata) {
            int dMax = ilog2ceil(n);
            int extended_n = 1 << dMax;
            int* dev_data;
            cudaMalloc((void**)&dev_data, sizeof(int) * extended_n);
            cudaMemset(dev_data, 0, sizeof(int) * extended_n);
            cudaMemcpy(dev_data, idata, sizeof(int) * n, cudaMemcpyHostToDevice);
            timer().startGpuTimer();
            // DONE
            scanInplace(extended_n, dev_data);
            timer().endGpuTimer();
            cudaMemcpy(odata, dev_data, sizeof(int) * n, cudaMemcpyDeviceToHost);
            cudaFree(dev_data);
        }

        __global__ void kernMapToBoolean(int n, int* bools, const int* idata) {
            int index = blockIdx.x * blockDim.x + threadIdx.x;
            if (index > n)return;
            if (idata[index] != 0)
                bools[index] = 1;
        }
        __global__ void kernScatter(int n, int* odata, const int* idata, const int* bools, const int* indices) {
            int index = blockIdx.x * blockDim.x + threadIdx.x;
            if (index > n)return;
            if (idata[index] != 0)
                odata[indices[index]] = idata[index];
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
        int compact(int n, int* odata, const int* idata) {
            int extended_n = 1 << ilog2ceil(n);
            int* dev_idata, * dev_odata, * dev_indices, num;
            cudaMalloc((void**)&dev_idata, sizeof(int) * extended_n);
            cudaMalloc((void**)&dev_odata, sizeof(int) * n);
            cudaMalloc((void**)&dev_indices, sizeof(int) * extended_n);
            cudaMemcpy(dev_idata, idata, sizeof(int) * n, cudaMemcpyHostToDevice);
            cudaMemset(dev_odata, 0, sizeof(int) * n);
            cudaMemset(dev_indices, 0, sizeof(int) * extended_n);
            dim3 fullBlocksPerGrid = ((n + blockSize - 1) / blockSize);
            timer().startGpuTimer();
            // DONE
            kernMapToBoolean << <fullBlocksPerGrid, blockSize >> > (n, dev_indices, dev_idata);
            scanInplace(extended_n, dev_indices);
            cudaMemcpy(&num, &dev_indices[extended_n - 1], sizeof(int), cudaMemcpyDeviceToHost);
            kernScatter << <fullBlocksPerGrid, blockSize >> > (n, dev_odata, dev_idata, dev_idata, dev_indices);
            timer().endGpuTimer();
            cudaMemcpy(odata, dev_odata, sizeof(int) * num, cudaMemcpyDeviceToHost);
            cudaFree(dev_idata);
            cudaFree(dev_odata);
            cudaFree(dev_indices);
            return num;
        }
    }
}

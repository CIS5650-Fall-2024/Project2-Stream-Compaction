#include <cuda.h>
#include <cuda_runtime.h>
#include "efficient.h"
#include "radixSort.h"

namespace StreamCompaction {
    namespace RadixSort {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        __global__ void kernMapToBoolean(int n, int* odata, const int* idata, int mask) {
            int index = blockIdx.x * blockDim.x + threadIdx.x;
            if (index >= n) {
                return;
            }
            odata[index] = (idata[index] & mask) == 0;
        }

        __global__ void kernScatter(int n, int* odata, const int* idata, const int* falses, int mask, int totalFalses) {
            int index = blockIdx.x * blockDim.x + threadIdx.x;
            if (index >= n) {
                return;
            }
            int falsesIndex = falses[index];
            if ((idata[index] & mask) == 0) {
                odata[falsesIndex] = idata[index];
            }
            else {
                odata[index - falsesIndex + totalFalses] = idata[index];
            }
        }

        void sort(int n, int* odata, const int* idata, int numBits) {
            int extended_n = 1 << ilog2ceil(n);
            dim3 fullBlocksPerGrid = BLOCKSPERGRID(n, blockSize);
            int* dev_idata;
            int* dev_odata;
            int* dev_falses;
            cudaMalloc((void**)&dev_idata, n * sizeof(int));
            cudaMalloc((void**)&dev_odata, n * sizeof(int));
            cudaMalloc((void**)&dev_falses, extended_n * sizeof(int));
            cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            timer().startGpuTimer();
            unsigned maxMask = 1 << numBits;
            for (unsigned mask = 1; mask < maxMask; mask <<= 1)
            {
                kernMapToBoolean << <fullBlocksPerGrid, blockSize >> > (n, dev_falses, dev_idata, mask);
                StreamCompaction::Efficient::scanInplace(extended_n, dev_falses);
                int totalFalses = 0, tmp_back = 0;
                cudaMemcpy(&totalFalses, dev_falses + n - 1, sizeof(int), cudaMemcpyDeviceToHost);
                cudaMemcpy(&tmp_back, dev_idata + n - 1, sizeof(int), cudaMemcpyDeviceToHost);
                totalFalses += (tmp_back & mask) == 0;
                kernScatter << <fullBlocksPerGrid, blockSize >> > (n, dev_odata, dev_idata, dev_falses, mask, totalFalses);
                std::swap(dev_idata, dev_odata);
            }
            timer().endGpuTimer();
            cudaMemcpy(odata, dev_idata, n * sizeof(int), cudaMemcpyDeviceToHost);
            cudaFree(dev_idata);
            cudaFree(dev_odata);
            cudaFree(dev_falses);
        }

        void sortShared(int n, int* odata, const int* idata, int numBits) {
            int extended_n = 1 << ilog2ceil(n);
            dim3 fullBlocksPerGrid = BLOCKSPERGRID(n, blockSize);
            int* dev_idata;
            int* dev_odata;
            cudaMalloc((void**)&dev_idata, n * sizeof(int));
            cudaMalloc((void**)&dev_odata, n * sizeof(int));
            Common::devDataBuffer buffer(extended_n, blockSize);
            cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            timer().startGpuTimer();
            unsigned int maxMask = 1 << numBits;
            for (unsigned mask = 1; mask < maxMask; mask <<= 1)
            {
                kernMapToBoolean << <fullBlocksPerGrid, blockSize >> > (n, buffer.data(), dev_idata, mask);
                StreamCompaction::Efficient::scanSharedInplace(extended_n, buffer);
                int totalFalses = 0, tmp_back = 0;
                cudaMemcpy(&totalFalses, buffer.data() + n - 1, sizeof(int), cudaMemcpyDeviceToHost);
                cudaMemcpy(&tmp_back, dev_idata + n - 1, sizeof(int), cudaMemcpyDeviceToHost);
                totalFalses += (tmp_back & mask) == 0;
                kernScatter << <fullBlocksPerGrid, blockSize >> > (n, dev_odata, dev_idata, buffer.data(), mask, totalFalses);
                std::swap(dev_idata, dev_odata);
            }
            timer().endGpuTimer();
            cudaMemcpy(odata, dev_idata, n * sizeof(int), cudaMemcpyDeviceToHost);
            cudaFree(dev_idata);
            cudaFree(dev_odata);
        }

    }
}

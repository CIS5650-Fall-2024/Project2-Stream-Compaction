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

        __global__ void kernUpSweep(int n, int d, int* data) {
            int index = threadIdx.x + (blockIdx.x * blockDim.x);
            int offset = 1 << (d + 1); // 2^(d+1)
            int pos = index * offset;

            if (pos >= n) {
                return;
            }

            data[pos + offset - 1] += data[pos + (offset >> 1) - 1];
        }

        __global__ void kernDownSweep(int n, int d, int* data) {
            int index = threadIdx.x + (blockIdx.x * blockDim.x);
            int offset = 1 << (d + 1); // 2^(d+1)
            int pos = index * offset;

            if (pos >= n) {
                return;
            }

            int t = data[pos + (offset >> 1) - 1];
            data[pos + (offset >> 1) - 1] = data[pos + offset - 1];
            data[pos + offset - 1] += t;
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            int enlargedSize = 1 << ilog2ceil(n); // enlarge the size to the nearest power of 2
            int* dev_idata;
            int blockSize = 64;

            cudaMalloc((void**)&dev_idata, enlargedSize * sizeof(int));
            checkCUDAError("cudaMalloc dev_idata failed!");

            // copy the input to GPU (size n data)
            cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);

            timer().startGpuTimer();
            // up-sweep
            for (int d = 0; d <= ilog2ceil(enlargedSize) - 1; d++) {
                int fullBlocksPerGrid = (enlargedSize / (1 << (d + 1)) + blockSize - 1) / blockSize;
                kernUpSweep<<<fullBlocksPerGrid, blockSize>>> (enlargedSize, d, dev_idata);
            }

            // down-sweep
            cudaMemset(dev_idata + enlargedSize - 1, 0, sizeof(int));
            for (int d = ilog2ceil(enlargedSize) - 1; d >= 0; d--) {
                int fullBlocksPerGrid = (enlargedSize / (1 << (d + 1)) + blockSize - 1) / blockSize;
                kernDownSweep<<<fullBlocksPerGrid, blockSize>>> (enlargedSize, d, dev_idata);
            }
            timer().endGpuTimer();

            // copy the result to odata (size n data)
            cudaMemcpy(odata, dev_idata, n * sizeof(int), cudaMemcpyDeviceToHost);

            // free memory
            cudaFree(dev_idata);
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
            int* dev_odata;
            int* dev_temp;
            int* dev_scan;
            int blockSize = 128;
            int fullBlocksPerGrid = (n + blockSize - 1) / blockSize;

            int enlargedSize = 1 << ilog2ceil(n); // enlarge the size to the nearest power of 2

            cudaMalloc((void**)&dev_idata, n * sizeof(int));
            checkCUDAError("cudaMalloc dev_idata failed!");
            cudaMalloc((void**)&dev_odata, n * sizeof(int));
            checkCUDAError("cudaMalloc dev_odata failed!");
            cudaMalloc((void**)&dev_temp, enlargedSize * sizeof(int));
            checkCUDAError("cudaMalloc dev_temp failed!");
            cudaMalloc((void**)&dev_scan, enlargedSize * sizeof(int));
            checkCUDAError("cudaMalloc dev_scan failed!");

            // copy the input to GPU
            cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);

            timer().startGpuTimer();
            // map
            StreamCompaction::Common::kernMapToBoolean<<<fullBlocksPerGrid, blockSize>>> (n, dev_temp, dev_idata);

            // scan (implemented again)
            cudaMemcpy(dev_scan, dev_temp, n * sizeof(int), cudaMemcpyDeviceToDevice);
            // up-sweep
            for (int d = 0; d <= ilog2ceil(enlargedSize) - 1; d++) {
                int fullBlocksPerGridEnlarged = (enlargedSize / (1 << (d + 1)) + blockSize - 1) / blockSize;
                kernUpSweep<<<fullBlocksPerGridEnlarged, blockSize>>> (enlargedSize, d, dev_scan);
            }

            // down-sweep
            cudaMemset(dev_scan + enlargedSize - 1, 0, sizeof(int));
            for (int d = ilog2ceil(enlargedSize) - 1; d >= 0; d--) {
                int fullBlocksPerGridEnlarged = (enlargedSize / (1 << (d + 1)) + blockSize - 1) / blockSize;
                kernDownSweep<<<fullBlocksPerGridEnlarged, blockSize>>> (enlargedSize, d, dev_scan);
            }

            // scatter
            StreamCompaction::Common::kernScatter<<<fullBlocksPerGrid, blockSize>>> (n, dev_odata, dev_idata, dev_temp, dev_scan);
            timer().endGpuTimer();

            // calculate count
            int lastScanValue = 0;
            int lastTempValue = 0;
            cudaMemcpy(&lastScanValue, dev_scan + n - 1, sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(&lastTempValue, dev_temp + n - 1, sizeof(int), cudaMemcpyDeviceToHost);
            int count = lastScanValue + lastTempValue;

            // copy the result to odata
            cudaMemcpy(odata, dev_odata, count * sizeof(int), cudaMemcpyDeviceToHost);

            // free memory
            cudaFree(dev_idata);
            cudaFree(dev_odata);
            cudaFree(dev_temp);
            cudaFree(dev_scan);

            return count;
        }
    }
}

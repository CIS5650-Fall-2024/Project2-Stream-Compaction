#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

#define blockSize 128

namespace StreamCompaction {
    namespace Efficient {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        __global__ void upSweep(int n, int d, int *data) {
            int index = threadIdx.x + (blockIdx.x * blockDim.x);
            if (index >= n) {
                return;
            }
            int pow2d = 1 << d;
            int pow2d1 = 1 << (d + 1);
            if (index % pow2d1 == 0) {
                data[index + pow2d1 - 1] += data[index + pow2d - 1];
            }
        }

        __global__ void downSweep(int n, int d, int *data) {
            int index = threadIdx.x + (blockIdx.x * blockDim.x);
            if (index >= n) {
                return;
            }
            int pow2d = 1 << d;
            int pow2d1 = 1 << (d + 1);
            if (index % pow2d1 == 0) {
                int t = data[index + pow2d - 1];
                data[index + pow2d - 1] = data[index + pow2d1 - 1];
                data[index + pow2d1 - 1] += t;
            }
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            int numIters = ilog2ceil(n);
            int nPowOf2 = 1 << numIters;

            int *dev_data;
            cudaMalloc((void**)&dev_data, nPowOf2 * sizeof(int));
            checkCUDAError("cudaMalloc failed");
            cudaMemcpy(dev_data, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            checkCUDAError("cudaMemcpy failed");
            int numBlocks = (nPowOf2 + blockSize - 1) / blockSize;

            timer().startGpuTimer();
            for (int d = 0; d < numIters; d++) {
                upSweep<<<numBlocks, blockSize>>>(nPowOf2, d, dev_data);
            }
            cudaMemset(dev_data + nPowOf2 - 1, 0, sizeof(int));
            for (int d = numIters - 1; d >= 0; d--) {
                downSweep<<<numBlocks, blockSize>>>(nPowOf2, d, dev_data);
            }
            timer().endGpuTimer();

            cudaMemcpy(odata, dev_data, n * sizeof(int), cudaMemcpyDeviceToHost);
            cudaFree(dev_data);
            checkCUDAError("cudaFree failed");
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
            int numIters = ilog2ceil(n);
            int nPowOf2 = 1 << numIters;

            int *dev_idata;
            int *dev_odata;
            int *dev_bools;
            int *dev_scanResult;
            cudaMalloc((void**)&dev_idata, n * sizeof(int));
            cudaMalloc((void**)&dev_odata, n * sizeof(int));
            cudaMalloc((void**)&dev_bools, n * sizeof(int));
            cudaMalloc((void**)&dev_scanResult, nPowOf2 * sizeof(int));
            checkCUDAError("cudaMalloc failed");
            cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            checkCUDAError("cudaMemcpy failed");
            int numBlocks = (nPowOf2 + blockSize - 1) / blockSize;

            timer().startGpuTimer();
            Common::kernMapToBoolean<<<numBlocks, blockSize>>>(n, dev_bools, dev_idata);
            Common::kernMapToBoolean<<<numBlocks, blockSize>>>(n, dev_scanResult, dev_idata);

            // NOTE: not calling scan() because don't want to double call timer().startCpuTimer()
            for (int d = 0; d < numIters; d++) {
                upSweep<<<numBlocks, blockSize>>>(nPowOf2, d, dev_scanResult);
            }
            cudaMemset(dev_scanResult + nPowOf2 - 1, 0, sizeof(int));
            for (int d = numIters - 1; d >= 0; d--) {
                downSweep<<<numBlocks, blockSize>>>(nPowOf2, d, dev_scanResult);
            }
            
            Common::kernScatter<<<numBlocks, blockSize>>>(n, dev_odata, dev_idata, dev_bools, dev_scanResult);

            int numElements = 0;
            cudaMemcpy(&numElements, dev_scanResult + nPowOf2 - 1, sizeof(int), cudaMemcpyDeviceToHost);
            if (n == nPowOf2 && idata[n - 1] != 0) {
                numElements++;
            }
            timer().endGpuTimer();

            cudaMemcpy(odata, dev_odata, numElements * sizeof(int), cudaMemcpyDeviceToHost);
            cudaFree(dev_idata);
            cudaFree(dev_odata);
            cudaFree(dev_bools);
            cudaFree(dev_scanResult);
            checkCUDAError("cudaFree failed");
            return numElements;
        }
    }
}

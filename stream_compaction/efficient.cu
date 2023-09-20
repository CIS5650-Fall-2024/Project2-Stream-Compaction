#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"
#include "device_launch_parameters.h"
#include <iostream>

#define BLOCK_SIZE 16

namespace StreamCompaction {
    namespace Efficient {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        __global__ void efficientScanUpSweep(int n, int nThread, int d, int *data) {
            int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
            if (idx >= nThread) return;
            int currIdx = ((idx + 1) << (d + 1)) - 1;
            int prevIdx = currIdx - (1 << d);
            data[currIdx] += data[prevIdx];
        }

        __global__ void efficientScanDownSweep(int n, int nThread, int d, int *data) {
            int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
            if (idx >= nThread) return;
            int currIdx = (n >> d) * (idx + 1) - 1;
            int prevIdx = currIdx - (n >> (d + 1));
            int temp = data[currIdx];
            data[currIdx] += data[prevIdx];
            data[prevIdx] = temp;
        }

        __global__ void replaceWithZero(int n, int nThread, int* data) {
            int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
            if (idx >= nThread) return;
            data[n - 1] = 0;
        }

        void efficientScanUpDownSweep(int n, int newN, int* dev_idata) {
            dim3 numBlocks;
            int nThread = newN;
            // up sweep
            for (int d = 0; d < ilog2ceil(n); d++) {
                nThread = newN >> (d + 1);
                numBlocks = (nThread + BLOCK_SIZE - 1) / BLOCK_SIZE;
                efficientScanUpSweep <<<numBlocks, BLOCK_SIZE >>> (newN, nThread, d, dev_idata);
            }
            // replace the last number of the array with 0.
            replaceWithZero <<<1, 1 >>> (newN, 1, dev_idata);
            // down sweep
            for (int d = 0; d < ilog2ceil(n); d++) {
                nThread = 1 << d;
                numBlocks = (nThread + BLOCK_SIZE - 1) / BLOCK_SIZE;
                efficientScanDownSweep <<<numBlocks, BLOCK_SIZE >>> (newN, nThread, d, dev_idata);
            }
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            int* dev_idata;
            int newN = 1 << ilog2ceil(n);
            cudaMalloc((void**)&dev_idata, sizeof(int) * newN);
            checkCUDAError("Efficient scan: cudaMalloc failed (dev_idata)");
            cudaMemcpy(dev_idata, idata, sizeof(int) * n, cudaMemcpyHostToDevice);
            timer().startGpuTimer();
            //// TODO
            efficientScanUpDownSweep(n, newN, dev_idata);

            timer().endGpuTimer();
            cudaMemcpy(odata, dev_idata, sizeof(int) * n, cudaMemcpyDeviceToHost);
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
            int newN = 1 << ilog2ceil(n);
            int* dev_idata;
            cudaMalloc((void**)&dev_idata, sizeof(int) * newN);
            checkCUDAError("Efficient scan: cudaMalloc failed (dev_idata)");
            int* dev_bools;
            cudaMalloc((void**)&dev_bools, sizeof(int) * newN);
            checkCUDAError("Efficient scan: cudaMalloc failed (dev_bools)");
            int* dev_indices;
            cudaMalloc((void**)&dev_indices, sizeof(int) * newN);
            checkCUDAError("Efficient scan: cudaMalloc failed (dev_indices)");

            cudaMemcpy(dev_idata, idata, sizeof(int) * n, cudaMemcpyHostToDevice);
            dim3 numBlocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
            timer().startGpuTimer();
            // TODO
            Common::kernMapToBoolean <<<numBlocks, BLOCK_SIZE>>> (n, dev_bools, dev_idata);
            cudaMemcpy(dev_indices, dev_bools, sizeof(int) * n, cudaMemcpyDeviceToDevice);

            efficientScanUpDownSweep(n, newN, dev_indices);

            Common::kernScatter <<<numBlocks, BLOCK_SIZE>>> (n, dev_bools, dev_idata, dev_bools, dev_indices);

            timer().endGpuTimer();
            int lastIdx;
            cudaMemcpy(&lastIdx, dev_indices + n - 1, sizeof(int), cudaMemcpyDeviceToHost);
            int lastBool;
            cudaMemcpy(&lastBool, dev_bools + n - 1, sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(odata, dev_bools, sizeof(int) * n, cudaMemcpyDeviceToHost);
            cudaFree(dev_idata);
            cudaFree(dev_bools);
            cudaFree(dev_indices);
            return lastIdx + lastBool;
        }
    }
}

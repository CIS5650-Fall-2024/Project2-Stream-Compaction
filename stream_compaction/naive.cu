#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"

#define BLOCK_SIZE 128

namespace StreamCompaction {
    namespace Naive {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }
        // TODO: __global__
        __global__ void naiveParallelScanAtLevelD(int n, int sumIdx, const int *idata, int* odata) {
            int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
            if (idx > n) return;
            if (idx >= sumIdx) {
                odata[idx] = idata[idx - sumIdx] + idata[idx];
            }
            else {
                odata[idx] = idata[idx];
            }
        }

        __global__ void include2exclude(int n, const int* idata, int* odata) {
            int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
            if (idx > n) return;
            if (idx == 0) {
                odata[idx] = 0;
            }
            else {
                odata[idx] = idata[idx - 1];
            }
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            // declear
            int *dev_idata, *dev_odata;
            // allocate memory on GPU
            cudaMalloc((void**)&dev_idata, sizeof(int) * n);
            checkCUDAError("Naive scan: cudaMalloc failed (dev_idata)");
            cudaMalloc((void**)&dev_odata, sizeof(int) * n);
            checkCUDAError("Naive scan: cudaMalloc failed (dev_odata)");
            // copy idata to GPU
            cudaMemcpy(dev_idata, idata, sizeof(int) * n, cudaMemcpyHostToDevice);
            dim3 numBlocks((n + BLOCK_SIZE - 1) / BLOCK_SIZE);
            timer().startGpuTimer();
            // TODO: call scan
            for (int d = 1; d <= ilog2ceil(n); d++) {
                naiveParallelScanAtLevelD <<<numBlocks, BLOCK_SIZE>>> (n, 1 << (d - 1), dev_idata, dev_odata);
                std::swap(dev_idata, dev_odata);
            }
            include2exclude <<<numBlocks, BLOCK_SIZE>>> (n, dev_idata, dev_odata);
            timer().endGpuTimer();
            // copy back odata to CPU
            cudaMemcpy(odata, dev_odata, sizeof(int) * n, cudaMemcpyDeviceToHost);
            // free memory
            cudaFree(dev_idata);
            cudaFree(dev_odata);
        }
    }
}

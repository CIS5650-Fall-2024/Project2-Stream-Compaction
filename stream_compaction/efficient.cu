#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

#include <iostream>

#define BLOCK_SIZE 128

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
            //int currIdx = (idx + 1) * powf(2, d + 1) - 1;
            //int prevIdx = currIdx - powf(2, d);
            data[currIdx] += data[prevIdx];
        }

        __global__ void efficientScanDownSweep(int n, int nThread, int d, int *data) {
            int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
            if (idx >= nThread) return;
            int currIdx = ((n * (idx + 1)) >> d) - 1;
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

        /*void printArr(int n, const int* data) {
            std::cout << "-------- print data --------" << std::endl;
            for (int i = 0; i < n; i++) {
                std::cout << data[i] << " ";
            }
            std::cout << std::endl;
        }*/

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            //printArr(n, idata);

            int* dev_idata;
            int newN = 1 << ilog2ceil(n);
            cudaMalloc((void**)&dev_idata, sizeof(int) * newN);
            //std::cout << "n before shift: " << nThread << std::endl;
            checkCUDAError("Efficient scan: cudaMalloc failed (dev_idata)");
            cudaMemcpy(dev_idata, idata, sizeof(int) * n, cudaMemcpyHostToDevice);
            timer().startGpuTimer();
            //// TODO
            //std::cout << "n after shift: " << nThread << std::endl;
            dim3 numBlocks;
            int nThread = newN;
            // up sweep
            for (int d = 0; d < ilog2ceil(n); d++) {
                nThread = newN >> (d + 1);
                //std::cout << "nthread: " << nThread << std::endl;
                numBlocks = (nThread + BLOCK_SIZE - 1) / BLOCK_SIZE;
                //std::cout << "nThread: " << nThread << " when d is " << d << std::endl;
                efficientScanUpSweep <<<numBlocks, BLOCK_SIZE>>> (newN, nThread, d, dev_idata);
                //cudaMemcpy(odata, dev_idata, sizeof(int) * n, cudaMemcpyDeviceToHost);
                //printArr(n, odata);
            }
            // replace the last number of the array with 0.
            replaceWithZero <<<1, 1>>> (newN, 1, dev_idata);
            //std::cout << "========= after replace =========" << std::endl;
            //cudaMemcpy(odata, dev_idata, sizeof(int) * n, cudaMemcpyDeviceToHost);
            //printArr(n, odata);
            // down sweep
            //std::cout << "========= down sweep =========" << std::endl;
            for (int d = 0; d < ilog2ceil(n); d++) {
                nThread = 1 << d;
                numBlocks = (nThread + BLOCK_SIZE - 1) / BLOCK_SIZE;
                efficientScanDownSweep <<<numBlocks, BLOCK_SIZE>>> (newN, nThread, d, dev_idata);
                //cudaMemcpy(odata, dev_idata, sizeof(int) * n, cudaMemcpyDeviceToHost);
                //printArr(n, odata);
            }
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
            timer().startGpuTimer();
            // TODO
            timer().endGpuTimer();
            return -1;
        }
    }
}

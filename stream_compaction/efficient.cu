#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
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

        //variable initialization
        int* dev_idata;
        int* dev_origin;
        const int blockSize = 64;

        //GPU scan
        __global__ void upSweep(int n, int logVal, int* idata) {
            int index = (blockIdx.x * blockDim.x) + threadIdx.x;

            if (index >= n)return;

            int offset1 = 1 << (logVal + 1);
            int offset2 = 1 << logVal;
            if (index % offset1 == 0) {
                idata[index + offset1 - 1] += idata[index + offset2 - 1];
            }
        }

        __global__ void downSweep(int n, int logVal, int* idata) {
            int index = (blockIdx.x * blockDim.x) + threadIdx.x;

            if (index >= n)return;

            int offset1 = 1 << (logVal + 1);
            int offset2 = 1 << logVal;

            if (index % offset1 == 0) {
                int t = idata[index + offset1 - 1];
                idata[index + offset1 - 1] += idata[index + offset2 - 1];
                idata[index + offset2 - 1] = t;
            }
        }

        __global__ void shiftArray(int n, int* origin, int* idata) {
            int index = (blockIdx.x * blockDim.x) + threadIdx.x;

            if (index >= n)return;

            idata[index] += origin[index];
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            int size = pow(2, ilog2ceil(n));

            cudaMalloc((void**)&dev_idata, size * sizeof(int));
            checkCUDAError("cudaMalloc dev_idata failed!");
            cudaMemset(dev_idata, 0, size * sizeof(int));
            checkCUDAError("cudaMemset dev_idata failed!");
            cudaMemcpy(dev_idata, idata, size * sizeof(int), cudaMemcpyHostToDevice);
            checkCUDAError("cudaMemcpy dev_idata failed!");

            cudaMalloc((void**)&dev_origin, size * sizeof(int));
            checkCUDAError("cudaMalloc dev_idata failed!");
            cudaMemset(dev_origin, 0, size * sizeof(int));
            checkCUDAError("cudaMemset dev_idata failed!");
            cudaMemcpy(dev_origin, idata, size * sizeof(int), cudaMemcpyHostToDevice);
            checkCUDAError("cudaMemcpy dev_idata failed!");

            dim3 fullBlocksPerGrid((n + blockSize - 1) / blockSize);
            timer().startGpuTimer();
            // TODO
            for (int i = 0; i <= ilog2ceil(n) - 1; i++) {
                upSweep << <fullBlocksPerGrid, blockSize >> > (size, i, dev_idata);
                checkCUDAError("cudaFunc upSweep failed!");
            }

            cudaMemset(dev_idata + size - 1, 0, sizeof(int));
            checkCUDAError("cudaMemset dev_idata + size - 1 failed!");

            for (int i = ilog2ceil(n) - 1; i >= 0; i--) {
                downSweep << <fullBlocksPerGrid, blockSize >> > (size, i, dev_idata);
                checkCUDAError("cudaFunc downSweep failed!");
            }

            shiftArray << <fullBlocksPerGrid, blockSize >> > (size, dev_origin, dev_idata);

            timer().endGpuTimer();
            cudaMemcpy(odata, dev_idata, sizeof(int) * n, cudaMemcpyDeviceToHost);
            cudaFree(dev_idata);
            cudaFree(dev_origin);
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

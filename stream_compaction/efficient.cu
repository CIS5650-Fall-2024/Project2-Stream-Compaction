#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"
#include <iostream>

namespace StreamCompaction {
    namespace Efficient {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        __global__ void upSweep(int n, int* data, int step) {
            int index = threadIdx.x + (blockIdx.x * blockDim.x);
            if (index < n && (index % step == 0)) {
                data[index + step - 1] += data[index + (step >> 1) - 1];
            }
        }

        __global__ void downSweep(int n, int* data, int step) {
            int index = threadIdx.x + (blockIdx.x * blockDim.x);
            if (index < n && (index % step == 0)) {
                int leftChildIndex = index + (step >> 1) - 1;
                int rightChildIndex = index + step - 1;
                int temp = data[leftChildIndex];
                data[leftChildIndex] = data[rightChildIndex];
                data[rightChildIndex] += temp;
            }
        }

        void printArray(const char* name, const int* array, int n) {
            std::cout << name << ": [";
            for (int i = 0; i < n; ++i) {
                std::cout << array[i];
                if (i < n - 1) {
                    std::cout << ", ";
                }
            }
            std::cout << "]" << std::endl;
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            int* dev_idata;
            dim3 fullBlocksPerGrid((n + blockSize - 1) / blockSize);
            int numLevels = ilog2ceil(n);
            int powerOf2Size = 1 << numLevels;
            int numZerosToSet = powerOf2Size - n;

            cudaMalloc((void**)&dev_idata, powerOf2Size * sizeof(int));
            checkCUDAError("cudaMalloc dev_idata failed!");
            cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            cudaMemset(&dev_idata[n], 0, numZerosToSet * sizeof(int));

            timer().startGpuTimer();
            // Step 1: Up Sweep
            for (int d = 0; d < numLevels; ++d) {
                upSweep <<<fullBlocksPerGrid, blockSize>>> (n, dev_idata, 1 << (d + 1));
            }

            // Step 2: Down Sweep
            cudaMemset(&dev_idata[powerOf2Size - 1], 0, sizeof(int));
            for (int d =  numLevels - 1; d >= 0; --d) {
                downSweep <<<fullBlocksPerGrid, blockSize>>> (n, dev_idata, 1 << (d + 1));
            }
            
            timer().endGpuTimer();
            cudaDeviceSynchronize();
            cudaMemcpy(odata, dev_idata, n * sizeof(int), cudaMemcpyDeviceToHost);
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

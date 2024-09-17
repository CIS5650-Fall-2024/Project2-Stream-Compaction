#include <cstdio>
#include <cuda.h>
#include <cuda_runtime.h>
#include "radix.h"

#include "common.h"
#include "efficient.h"

#define blockSize 128

namespace StreamCompaction {
    namespace Radix {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        __global__ void kernMapToCurrBits(int n, int *odata, const int *idata, int bit) {
            int index = threadIdx.x + (blockIdx.x * blockDim.x);
            if (index >= n) {
                return;
            }

            // get the bit-th bit of the number, but also flip the bit to produce the e array
            odata[index] = !((idata[index] >> bit) & 1);
        }

        __global__ void kernComputeT(int n, int *odata, const int *idata, const int totalFalses) {
            int index = threadIdx.x + (blockIdx.x * blockDim.x);
            if (index >= n) {
                return;
            }

            odata[index] = index - idata[index] + totalFalses;
        }

        __global__ void kernComputeD(int n, int *odata, const int *be, const int *t, const int *f) {
            int index = threadIdx.x + (blockIdx.x * blockDim.x);
            if (index >= n) {
                return;
            }

            // be array is stored as e array. To access b, reverse the condition
            odata[index] = be[index] ? f[index] : t[index];
        }

        __global__ void scatter(int n, int *odata, const int *idata, const int *d) {
            int index = threadIdx.x + (blockIdx.x * blockDim.x);
            if (index >= n) {
                return;
            }

            odata[d[index]] = idata[index];
        }

        int getMaxBits(int n, const int *idata) {
            // find maximum number in the array
            int maxNum = 0;
            for (int i = 0; i < n; i++) {
                maxNum = std::max(maxNum, idata[i]);
            }
            // calculate the number of bits of the maximum number
            int maxBits = 0;
            while (maxNum > 0) {
                maxNum >>= 1;
                maxBits++;
            }
            return maxBits;
        }

        /**
         * Performs radix sort on idata, storing the result into odata.
         * @param n the number of elements in idata
         * @param odata output.txt data
         * @param idata input data
         * @param maxBits the maximum number of bits of a given number in the array
         */
        void sort(int n, int *odata, const int *idata, int maxBits) {
            int* dev_idata;
            int* dev_odata;
            int* dev_be;
            int* dev_f;
            int* dev_t;
            int* dev_d;
            int fullBlocksPerGrid = (n + blockSize - 1) / blockSize;

            cudaMalloc((void**)&dev_idata, n * sizeof(int));
            checkCUDAError("cudaMalloc dev_idata failed!");
            cudaMalloc((void**)&dev_odata, n * sizeof(int));
            checkCUDAError("cudaMalloc dev_odata failed!");
            cudaMalloc((void**)&dev_be, n * sizeof(int));
            checkCUDAError("cudaMalloc dev_b failed!");
            cudaMalloc((void**)&dev_f, n * sizeof(int));
            checkCUDAError("cudaMalloc dev_f failed!");
            cudaMalloc((void**)&dev_t, n * sizeof(int));
            checkCUDAError("cudaMalloc dev_t failed!");
            cudaMalloc((void**)&dev_d, n * sizeof(int));
            checkCUDAError("cudaMalloc dev_d failed!");

            // copy the input to GPU (size n data)
            cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);

            timer().startGpuTimer();
            for (int i = 0; i < maxBits; i++) {
                // b & e array (b can be acquired by flipping back)
                kernMapToCurrBits<<<fullBlocksPerGrid, blockSize>>>(n, dev_be, dev_idata, i);
                // exclusive scan f array
                StreamCompaction::Efficient::scan(n, dev_f, dev_be);
                // calculate totalFalses
                int lastFVal = 0;
                int lastEVal = 0;
                cudaMemcpy(&lastEVal, dev_be + n - 1, sizeof(int), cudaMemcpyDeviceToHost);
                cudaMemcpy(&lastFVal, dev_f + n - 1, sizeof(int), cudaMemcpyDeviceToHost);
                int totalFalses = lastFVal + lastEVal;
                // t array
                kernComputeT<<<fullBlocksPerGrid, blockSize>>>(n, dev_t, dev_f, totalFalses);
                // d array
                kernComputeD<<<fullBlocksPerGrid, blockSize>>>(n, dev_d, dev_be, dev_t, dev_f);
                // scatter
                scatter<<<fullBlocksPerGrid, blockSize>>>(n, dev_odata, dev_idata, dev_d);
                // swap
                std::swap(dev_idata, dev_odata);
            }
            timer().endGpuTimer();

            // copy the result to odata (size n data)
            cudaMemcpy(odata, dev_idata, n * sizeof(int), cudaMemcpyDeviceToHost);

            cudaFree(dev_idata);
            cudaFree(dev_odata);
            cudaFree(dev_be);
            cudaFree(dev_f);
            cudaFree(dev_t);
            cudaFree(dev_d);
        }
    }
}
#include <cuda.h>
#include <cuda_runtime.h>
#include <algorithm>
#include "common.h"
#include "radix.h"

#define blockSize 256

void printArrayT(int n, int* a, bool abridged = false) {
    printf("    [ ");
    for (int i = 0; i < n; i++) {
        if (abridged && i + 2 == 15 && n > 16) {
            i = n - 2;
            printf("... ");
        }
        printf("%3d ", a[i]);
    }
    printf("]\n");
}
namespace StreamCompaction {
    namespace Radix {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        __global__ void kernBitMaskNot(int n, int digit, int* odata, int* idata) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= n) return;

            int bitmask = 1 << digit;

            odata[idx] = (int)(!(bool)(idata[idx] & bitmask));
        }

        __global__ void kernScatter(int n, int fCount, int* odata_scat, int* f, int* e, int* idata) {
            // odata = d array, idata = i array
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= n) return;

            int actualIdx = !e[idx] ? idx - f[idx] + fCount : f[idx];
            odata_scat[actualIdx] = idata[idx];
        }

        // copied from naive.cu
        __global__ void kernScan(int n, int depth, int* odata, int* idata) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= n) return;

            if (idx < depth) {
                odata[idx] = idata[idx];
                return;
            }

            odata[idx] = idata[idx - depth] + idata[idx];
            return;
        }

        __global__ void kernToExclusive(int n, int* odata, int* idata) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= n) return;

            if (idx == 0) {
                odata[idx] = 0;
            }
            else {
                odata[idx] = idata[idx - 1];
            }
            return;
        }

        /**
         * Performs radixSort on idata, storing the result into odata.
         */
        void radixSort(int n, int *odata, const int *idata) {
            int* dev_odata;
            int* dev_idata;

            int* dev_odata_scat;
            int* dev_idata_const;

            int* dev_temp;

            cudaMalloc((void**)&dev_odata, n * sizeof(int));
            checkCUDAError("cudaMalloc dev_odata failed");
            cudaMalloc((void**)&dev_idata, n * sizeof(int));
            checkCUDAError("cudaMalloc dev_idata failed");

            cudaMalloc((void**)&dev_odata_scat, n * sizeof(int));
            checkCUDAError("cudaMalloc dev_idata_const failed");
            cudaMalloc((void**)&dev_idata_const, n * sizeof(int));
            checkCUDAError("cudaMalloc dev_idata_const failed");

            cudaMalloc((void**)&dev_temp, n * sizeof(int));
            checkCUDAError("cudaMalloc dev_idata_const failed");

            cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            checkCUDAError("cudaMemcpy to dev_idata failed");
            cudaMemcpy(dev_idata_const, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            checkCUDAError("cudaMemcpy to dev_idata_const failed");

            int* test = new int[n];

            // find the maximum number (to set the number of iterations of bitmask)
            int maxDigitLen = 31;
            /*for (int i = 0; i < n; i++) {
                maxDigitLen = std::max(maxDigitLen, ilog2ceil(idata[i]));
            }*/

            if (time)
                timer().startGpuTimer();
            // TODO

            // upsweep
            for (int d = 0; d < maxDigitLen; d++) {
                dim3 blocksPerGrid((n + blockSize - 1) / blockSize);

                // produce 'e' array to dev_odata
                kernBitMaskNot<<<blocksPerGrid, blockSize>>>(n, d, dev_odata, dev_idata_const);

                int fCount, tempCount;
                cudaMemcpy(&tempCount, dev_odata + n - 1, sizeof(int), cudaMemcpyDeviceToHost);
                std::swap(dev_idata, dev_odata); // store 'e' to dev_idata

                // exclusive scan to get 'f' array to dev_odata (copied from naive.cu)
                kernScan<<<blocksPerGrid, blockSize >> > (n, 1, dev_temp, dev_idata);
                for (int depth = 2; depth <= ilog2ceil(n); ++depth) {
                    kernScan<<<blocksPerGrid, blockSize>>>(n, 1 << (depth - 1), dev_odata, dev_temp);
                    std::swap(dev_odata, dev_temp);
                }
                kernToExclusive<<<blocksPerGrid, blockSize>>>(n, dev_odata, dev_temp);

                cudaMemcpy(&fCount, dev_odata + n - 1, sizeof(int), cudaMemcpyDeviceToHost);
                fCount += tempCount;

                // scatter to get 'f' and 'e' arrays
                kernScatter<<<blocksPerGrid, blockSize>>>(n, fCount, dev_odata_scat, dev_odata, dev_idata, dev_idata_const);

                // pass scatter result to next iteration
                std::swap(dev_odata_scat, dev_idata_const);
            }

            if (time)
                timer().endGpuTimer();

            std::swap(dev_odata_scat, dev_idata_const);

            cudaMemcpy(odata, dev_odata_scat, n * sizeof(int), cudaMemcpyDeviceToHost);
            checkCUDAError("cudaMemcpy from dev_odata_scat failed");

            cudaFree(dev_odata);
            checkCUDAError("cudaFree dev_odata failed");
            cudaFree(dev_idata);
            checkCUDAError("cudaFree dev_idata failed");
            cudaFree(dev_odata_scat);
            checkCUDAError("cudaFree dev_odata_scat failed");
            cudaFree(dev_idata_const);
            checkCUDAError("cudaFree dev_idata_const failed");
            cudaFree(dev_temp);
            checkCUDAError("cudaFree dev_temp failed");
        }
    }
}

#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

#include <iostream>

#define blockSize 256

#define DEBUG_PRINT 0
#define DEBUG_PRINT_ARRAYS 0

#define CONVERT_TO_INCLUSIVE 0

namespace StreamCompaction {
    namespace Efficient {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        __global__ void kernZeroArray(const int n, int* data) {
            int k = (blockIdx.x * blockDim.x) + threadIdx.x;
            if (k < n) {
                data[k] = 0;
            }
        }

        __global__ void kernUpSweepStep(const int n, const int d, int* data) {
            int twoD = 1 << d;
            int twoDPlusOne = twoD << 1;
            
            // Really not sure why, but changing this from int to size_t avoids an illegal memory address error
            // for arrays of size >= 2^24. I don't think k should be exceeding the maximum integer value in
            // any case since the number of threads is equal to nPadded/twoDPlusOne, so k should be no more than
            // nPadded, which is an int.
            size_t k = twoDPlusOne * ((blockIdx.x * blockDim.x) + threadIdx.x);
            if (k >= n) {
                return;
            }

            data[k + twoDPlusOne - 1] += data[k + twoD - 1];
        }

        __global__ void kernSetLastElementToZero(const int n, int* data) {
            data[n - 1] = 0;
        }

        __global__ void kernDownSweepStep(const int n, const int d, int* data) {
            int twoD = 1 << d;
            int twoDPlusOne = twoD << 1;

            size_t k = twoDPlusOne * ((blockIdx.x * blockDim.x) + threadIdx.x);
            if (k >= n) {
                return;
            }

            int t = data[k + twoD - 1];
            data[k + twoD - 1] = data[k + twoDPlusOne - 1];
            data[k + twoDPlusOne - 1] += t;
        }

        __global__ void kernExclusiveToInclusive(const int n, const int* idata, const int lastNumber, int* odata) {
            int k = (blockIdx.x * blockDim.x) + threadIdx.x;
            if (k >= n) {
                return;
            }

            if (k < n - 1) {
                odata[k] = idata[k + 1];
            } else {
                odata[k] = idata[k] + lastNumber;
            }
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            const int lastNumber = idata[n - 1];
            const int log2ceil_n = ilog2ceil(n);
            const int nPadded = 1 << log2ceil_n;

            int* dev_data1;
            int* dev_data2;

            cudaMalloc((void**)&dev_data1, nPadded * sizeof(int));
            checkCUDAErrorFn("cudaMalloc dev_data1 failed!");
            cudaMalloc((void**)&dev_data2, nPadded * sizeof(int));
            checkCUDAErrorFn("cudaMalloc dev_data2 failed!");

            cudaMemset(dev_data1, 0, nPadded * sizeof(int));
            checkCUDAErrorFn("cudaMemset dev_data1 failed!");

            cudaMemcpy(dev_data1, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            checkCUDAErrorFn("cudaMempcy idata to dev_data1 failed!");

            timer().startGpuTimer();

#if DEBUG_PRINT
            printf("\n================================\n");
            printf("UP SWEEP\n");
            printf("================================\n\n");

            cudaMemcpy(odata, dev_data1, n * sizeof(int), cudaMemcpyDeviceToHost);
            checkCUDAErrorFn("cudaMempcy to host failed!");
#if DEBUG_PRINT_ARRAYS
            for (int i = 0; i < n; ++i) {
                std::cout << odata[i] << ", ";
            }
            std::cout << std::endl;
#endif
#endif

            for (int d = 0; d < log2ceil_n; ++d) {
                int numThreads = nPadded / (1 << (d + 1));
                dim3 fullBlocksPerGrid((numThreads + blockSize - 1) / blockSize);

#if DEBUG_PRINT
                std::cout << "d = " << d << std::endl;
#endif

                kernUpSweepStep<<<fullBlocksPerGrid, blockSize>>>(nPadded, d, dev_data1);

#if DEBUG_PRINT
                cudaMemcpy(odata, dev_data1, n * sizeof(int), cudaMemcpyDeviceToHost);
                checkCUDAErrorFn("cudaMempcy to host failed!");
#if DEBUG_PRINT_ARRAYS
                for (int i = 0; i < n; ++i) {
                    std::cout << odata[i] << ", ";
                }
                std::cout << std::endl;
#endif
#endif
            }

            kernSetLastElementToZero<<<1, 1>>>(nPadded, dev_data1);

#if DEBUG_PRINT
            printf("\n================================\n");
            printf("DOWN SWEEP\n");
            printf("================================\n\n");

            cudaMemcpy(odata, dev_data1, n * sizeof(int), cudaMemcpyDeviceToHost);
            checkCUDAErrorFn("cudaMempcy to host failed!");
#if DEBUG_PRINT_ARRAYS
            for (int i = 0; i < n; ++i) {
                std::cout << odata[i] << ", ";
            }
            std::cout << std::endl;
#endif
#endif

            for (int d = log2ceil_n - 1; d >= 0; --d) {
                int numThreads = nPadded / (1 << (d + 1));
                dim3 fullBlocksPerGrid((numThreads + blockSize - 1) / blockSize);

#if DEBUG_PRINT
                std::cout << "d = " << d << std::endl;
#endif

                kernDownSweepStep<<<fullBlocksPerGrid, blockSize>>>(nPadded, d, dev_data1);

#if DEBUG_PRINT
                cudaMemcpy(odata, dev_data1, n * sizeof(int), cudaMemcpyDeviceToHost);
                checkCUDAErrorFn("cudaMempcy to host failed!");
                for (int i = 0; i < n; ++i) {
                    std::cout << odata[i] << ", ";
                }
                std::cout << std::endl;
#endif
            }

#if CONVERT_TO_INCLUSIVE
#if DEBUG_PRINT
            printf("\n================================\n");
            printf("CONVERT TO INCLUSIVE\n");
            printf("================================\n\n");
#endif

            dim3 fullBlocksPerGrid((n + blockSize - 1) / blockSize);
            kernExclusiveToInclusive<<<fullBlocksPerGrid, blockSize>>>(n, dev_data1, lastNumber, dev_data2);
            std::swap(dev_data1, dev_data2);

#if DEBUG_PRINT
            cudaMemcpy(odata, dev_data1, n * sizeof(int), cudaMemcpyDeviceToHost);
            checkCUDAErrorFn("cudaMempcy to host failed!");
#if DEBUG_PRINT_ARRAYS
            for (int i = 0; i < n; ++i) {
                std::cout << odata[i] << ", ";
            }
            std::cout << std::endl;
#endif
#endif
#endif

            timer().endGpuTimer();

            cudaMemcpy(odata, dev_data1, n * sizeof(int), cudaMemcpyDeviceToHost);
            checkCUDAErrorFn("cudaMempcy dev_data2 to odata failed!");

            cudaFree(dev_data1);
            cudaFree(dev_data2);
            checkCUDAErrorFn("cudaFree failed!");
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
            const int log2ceil_n = ilog2ceil(n);
            const int nPadded = 1 << log2ceil_n;

            int* dev_idata;
            int* dev_bools;
            int* dev_scan;
            int* dev_odata;

            cudaMalloc((void**)&dev_idata, n * sizeof(int));
            checkCUDAErrorFn("cudaMalloc dev_idata failed!");
            cudaMalloc((void**)&dev_bools, n * sizeof(int));
            checkCUDAErrorFn("cudaMalloc dev_bools failed!");
            cudaMalloc((void**)&dev_scan, nPadded * sizeof(int));
            checkCUDAErrorFn("cudaMalloc dev_scan failed!");
            cudaMalloc((void**)&dev_odata, n * sizeof(int));
            checkCUDAErrorFn("cudaMalloc dev_odata failed!");

            cudaMemset(dev_scan, 0, nPadded * sizeof(int));
            checkCUDAErrorFn("cudaMemset dev_scan failed!");

            cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            checkCUDAErrorFn("cudaMempcy idata to dev_idata failed!");

            dim3 fullBlocksPerGridConst((n + blockSize - 1) / blockSize);

            timer().startGpuTimer();

            StreamCompaction::Common::kernMapToBoolean<<<fullBlocksPerGridConst, blockSize>>>(n, dev_bools, dev_idata);

            cudaMemcpy(dev_scan, dev_bools, n * sizeof(int), cudaMemcpyDeviceToDevice);

            for (int d = 0; d < log2ceil_n; ++d) {
                int numThreads = nPadded / (1 << (d + 1));
                dim3 fullBlocksPerGrid((numThreads + blockSize - 1) / blockSize);
                kernUpSweepStep<<<fullBlocksPerGrid, blockSize>>>(nPadded, d, dev_scan);
            }

            kernSetLastElementToZero<<<1, 1>>>(nPadded, dev_scan);

            for (int d = log2ceil_n - 1; d >= 0; --d) {
                int numThreads = nPadded / (1 << (d + 1));
                dim3 fullBlocksPerGrid((numThreads + blockSize - 1) / blockSize);
                kernDownSweepStep<<<fullBlocksPerGrid, blockSize>>>(nPadded, d, dev_scan);
            }

            int numRemaining;
            cudaMemcpy(&numRemaining, dev_scan + (nPadded - 1), 1 * sizeof(int), cudaMemcpyDeviceToHost);
            checkCUDAErrorFn("cudaMempcy numRemaining to host failed!");

            StreamCompaction::Common::kernScatter<<<fullBlocksPerGridConst, blockSize>>>(n, dev_odata, dev_idata, dev_bools, dev_scan);

            timer().endGpuTimer();

            cudaMemcpy(odata, dev_odata, n * sizeof(int), cudaMemcpyDeviceToHost);
            checkCUDAErrorFn("cudaMempcy dev_odata to odata failed!");

            cudaFree(dev_idata);
            cudaFree(dev_bools);
            cudaFree(dev_scan);
            cudaFree(dev_odata);
            checkCUDAErrorFn("cudaFree failed!");

            return numRemaining;
        }
    }
}

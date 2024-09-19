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

        __global__ void kernUpSweep(int n, int* A, int offset) {
            int idx = blockDim.x * blockIdx.x + threadIdx.x;

            // Dividing the input into groups each of offset size
            if (idx >= n / offset) {
                return;
            }
            idx *= offset;
            A[idx + offset - 1] += A[idx + offset / 2 - 1];
        }

        __global__ void kernDownSweep(int n, int* A, int offset) {
            int idx = blockDim.x * blockIdx.x + threadIdx.x;
            if (idx >= n/offset) {
                return;
            }
            idx *= offset;

            int temp = A[idx + offset / 2 - 1];
            A[idx + offset / 2 - 1] = A[idx + offset - 1];
            A[idx + offset - 1] += temp;
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int* odata, const int* idata, bool timeFlag) {

            unsigned int blockSize = 128;

            int padding = 1 << ilog2ceil(n);

            int* dev_odata;
            size_t arraySize = n * sizeof(int);
            size_t paddedSize = padding * sizeof(int);
            cudaMalloc((void**)&dev_odata, paddedSize);
            checkCUDAError("cudaMalloc dev_odata failed!");

            cudaMemcpy(dev_odata, idata, arraySize, cudaMemcpyHostToDevice);
            checkCUDAError("cudaMemcpy idata to dev_odata failed!");

            cudaMemset(dev_odata + n, 0, (paddedSize - arraySize));
            checkCUDAError("cudaMemcpy padding dev_odata failed!");

            int numThreads = padding;

            if (timeFlag)
                timer().startGpuTimer();
            for (int i = 0; i < ilog2ceil(n); i++) {
                int offset = 1 << (i + 1);
                numThreads /= 2;
                
                dim3 fullBlocksPerGrid = ((numThreads + blockSize - 1) / blockSize);
                kernUpSweep << <fullBlocksPerGrid, blockSize >> > (padding, dev_odata, offset);
                cudaDeviceSynchronize();
                checkCUDAError("kernUpSweep failed!");
            }

            // assign 0 to the root of the tree for Down-Sweep
            cudaMemset(dev_odata + padding - 1, 0, sizeof(int));
            cudaDeviceSynchronize();
            checkCUDAError("cudaMemset to dev_odata failed!");

            for (int i = ilog2ceil(n) - 1; i >= 0; i--) {
                int offset = 1 << (i + 1);
                numThreads *= 2;
                dim3 fullBlocksPerGrid = ((numThreads + blockSize - 1) / blockSize);
                kernDownSweep << <fullBlocksPerGrid, blockSize >> > (padding, dev_odata, offset);
                checkCUDAError("kernDownSweep failed!");
            }
            if (timeFlag)
                timer().endGpuTimer();

            cudaMemcpy(odata, dev_odata, arraySize, cudaMemcpyDeviceToHost);
            cudaDeviceSynchronize();
            checkCUDAError("cudaMemcpy dev_odata to odata failed!");

            cudaFree(dev_odata);
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
        int compact(int n, int* odata, const int* idata) {

            // Create device arrays
            int* dev_idata;
            int* dev_odata;
            int* dev_bools;
            int* dev_indices;
            int padding = 1 << ilog2ceil(n);
            size_t arraySize = n * sizeof(int);
            size_t paddedSize = padding * sizeof(int);

            cudaMalloc((void**)&dev_idata, arraySize);
            cudaDeviceSynchronize();
            checkCUDAError("cudaMalloc1 failed!");

            cudaMalloc((void**)&dev_bools, paddedSize);
            cudaDeviceSynchronize();
            checkCUDAError("cudaMalloc2 failed!");

            cudaMalloc((void**)&dev_indices, paddedSize);
            cudaMalloc((void**)&dev_odata, arraySize);
            cudaDeviceSynchronize();
            checkCUDAError("cudaMalloc failed!");

            cudaMemcpy(dev_idata, idata, arraySize, cudaMemcpyHostToDevice);
            checkCUDAError("cudaMemcpy idata failed!");

            cudaMemset(dev_bools + n, 0, (paddedSize - arraySize));
            checkCUDAError("cudaMemset dev_bools failed!");

            unsigned int blockSize = 128;
            dim3 fullBlocksPerGrid((n + blockSize - 1) / blockSize);

            timer().startGpuTimer();

            StreamCompaction::Common::kernMapToBoolean << <fullBlocksPerGrid, blockSize >> > (n, dev_bools, dev_idata);

            scan(n, dev_indices, dev_bools, 0);
            StreamCompaction::Common::kernScatter << <fullBlocksPerGrid, blockSize >> > (n, dev_odata, dev_idata, dev_bools, dev_indices);
            cudaDeviceSynchronize();
            checkCUDAError("kernel calls failed!");

            timer().endGpuTimer();
            cudaMemcpy(odata, dev_odata, arraySize, cudaMemcpyDeviceToHost);
            checkCUDAError("cudaMemcpy dev_odata to odata failed!");

            // check if last element of idata is valid, by checking dev_bools. 
            // If yes, then its index is compactLen - 1. If not, its index is compactLen.
            int isLastElemValid;
            cudaMemcpy(&isLastElemValid, dev_bools + n - 1, sizeof(int), cudaMemcpyDeviceToHost);

            int lastElemIdx;
            cudaMemcpy(&lastElemIdx, dev_indices + n - 1, sizeof(int), cudaMemcpyDeviceToHost);

            int compactLen = (isLastElemValid) ? lastElemIdx + 1 : lastElemIdx;

            cudaFree(dev_bools);
            cudaFree(dev_indices);
            cudaFree(dev_idata);
            cudaFree(dev_odata);

            return (compactLen) ? compactLen : -1;
        }
    }
}

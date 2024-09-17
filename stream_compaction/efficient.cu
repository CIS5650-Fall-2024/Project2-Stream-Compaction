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
            idx *= offset;
            if (idx >= n) {
                return;
            }
            A[idx + offset - 1] = A[idx + offset / 2 - 1] + A[idx + offset / 2];
        }

        __global__ void kernDownSweep(int n, int* A, int offset) {
            int idx = blockDim.x * blockIdx.x + threadIdx.x;
            if (idx >= n) {
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
        void scan(int n, int *odata, const int *idata) {
            timer().startGpuTimer();
            unsigned int blockSize = 128;
            
            int* A;
            size_t arraySize = n * sizeof(int);
            cudaMalloc((void**)&A, arraySize);
            checkCUDAError("cudaMalloc A failed!");

            cudaMemcpy(A, idata, arraySize, cudaMemcpyHostToDevice);
            checkCUDAError("cudaMemcpy idata to A failed!");

            for (int i = 0; i < ilog2ceil(n); i++) {
                int offset = pow(2, (i + 1));
                blockSize /= offset;
                dim3 fullBlocksPerGrid = ((n + blockSize - 1) / blockSize);
                kernUpSweep << <fullBlocksPerGrid, blockSize >> > (n, A, offset);
            }

            A[n - 1] = 0;
            for (int i = ilog2ceil(n) - 1; i >= 0; i++) {
                int offset = pow(2, i + 1);
                blockSize /= offset;
                dim3 fullBlocksPerGrid = ((n + blockSize - 1) / blockSize);
                kernDownSweep << <fullBlocksPerGrid, blockSize >> > (n, A, offset);
            }

            cudaMemcpy(odata, A, arraySize, cudaMemcpyDeviceToHost);
            checkCUDAError("cudaMemcpy odata failed!");

            cudaFree(A);

            timer().endGpuTimer();
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

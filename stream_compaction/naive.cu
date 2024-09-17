#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h" 
#include "naive.h"

namespace StreamCompaction {
    namespace Naive {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        __global__ void kernNaiveScan(int n, int *B, int *A, int d) {
            int idx = blockDim.x * blockIdx.x + threadIdx.x;

            if (idx >= n) {
                return;
            }
            int offset = 1 << (d - 1);

            if (idx >= offset) {
                B[idx] = A[idx - offset] + A[idx];
            }
            else {
                B[idx] = A[idx];
            }
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) { 
            timer().startGpuTimer();
            // TODO
            unsigned int blockSize = 128;
            dim3 fullBlocksPerGrid((n + blockSize - 1) / blockSize);
            
            // define two device arrays
            int* A;
            int* B;

            size_t arraySize = n * sizeof(int);
            cudaMalloc((void**)&A, arraySize);
            checkCUDAError("cudaMalloc A failed!");
            cudaMalloc((void**)&B, arraySize);
            checkCUDAError("cudaMalloc B failed!");

            cudaMemcpy(A, idata, arraySize, cudaMemcpyHostToDevice);
            checkCUDAError("cudaMemcpy idata to A failed!");

            for (int i = 1; i <= ilog2ceil(n); i++) {
                kernNaiveScan << <fullBlocksPerGrid, blockSize >> > (n, B, A, i);

                // swap the two arrays
                std::swap(B, A);
            }

            odata[0] = 0;
            cudaMemcpy(odata + 1, A, arraySize, cudaMemcpyDeviceToHost);
            checkCUDAError("cudaMemcpy odata failed!");

            cudaFree(A);
            cudaFree(B);

            timer().endGpuTimer();
        }
    }
}

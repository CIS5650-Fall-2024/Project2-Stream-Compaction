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

        // TODO: __global__

        __global__ void kernNaiveScan(int n, int offset, int* x, const int* last) {
            int index = threadIdx.x + (blockIdx.x * blockDim.x);
            if (index >= n) {
				return;
			}

            x[index] = last[index];
            if (index >= offset) {
                x[index] += last[index - offset];
            }
        }

        __global__ void kernShift(int n, int* x, const int* last) {
            int index = threadIdx.x + (blockIdx.x * blockDim.x);
            if (index >= n) {
                return;
            }

            x[index] = index == 0 ? 0 : last[index - 1];
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            // memory operation
            int *x, *last;
            cudaMalloc((void**)&x, n * sizeof(int));
            cudaMalloc((void**)&last, n * sizeof(int));
            cudaMemcpy(last, idata, n * sizeof(int), cudaMemcpyHostToDevice);

            timer().startGpuTimer();

            int blockSize = 64;
            dim3 fullBlocksPerGrid((n + blockSize - 1) / blockSize);

            int max_d = ilog2ceil(n);
            // TODO
            for (int d = 1; d <= max_d; ++d) {
                kernNaiveScan<<<fullBlocksPerGrid, blockSize >>>(n, pow(2, d - 1), x, last);
				std::swap(x, last);
			}

            kernShift<<<fullBlocksPerGrid, blockSize >>>(n, x, last);

            timer().endGpuTimer();

            cudaMemcpy(odata, x, n * sizeof(int), cudaMemcpyDeviceToHost);
            cudaFree(x);
            cudaFree(last);
        }
    }
}

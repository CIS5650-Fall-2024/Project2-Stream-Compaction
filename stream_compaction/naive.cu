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

        __global__ void kernNaiveScan(int n, int d, int* odata, const int* idata) {
            int index = threadIdx.x + (blockIdx.x * blockDim.x);
            if (index >= n) {
                return;
            }

            int offset = 1 << (d - 1);
            if (index >= offset) {
                odata[index] = idata[index - offset] + idata[index];
            }
            else {
                odata[index] = idata[index];
            }
        }

        void ChangeToExclusive(int n, int* odata) {
            for (int i = n - 1; i > 0; i--) {
                odata[i] = odata[i - 1];
            }
            odata[0] = 0;
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            int* dev_idata;
            int* dev_odata;
            int blockSize = 128;
            int fullBlocksPerGrid = (n + blockSize - 1) / blockSize;

            cudaMalloc((void**)&dev_idata, n * sizeof(int));
            checkCUDAError("cudaMalloc dev_idata failed!");
            cudaMalloc((void**)&dev_odata, n * sizeof(int));
            checkCUDAError("cudaMalloc dev_odata failed!");

            // copy the input to GPU
            cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);

            timer().startGpuTimer();
            // outer loop
            for (int d = 1; d <= ilog2ceil(n); d++) {
                // parallel process
                kernNaiveScan<<<fullBlocksPerGrid, blockSize>>> (n, d, dev_odata, dev_idata);
                std::swap(dev_odata, dev_idata);
            }
            timer().endGpuTimer();

            cudaMemcpy(odata, dev_idata, n * sizeof(int), cudaMemcpyDeviceToHost);

            // shift right and add 0 to the beginning to acquire the exclusive scan
            ChangeToExclusive(n, odata);

            // free memory
            cudaFree(dev_idata);
            cudaFree(dev_odata);
        }
    }
}

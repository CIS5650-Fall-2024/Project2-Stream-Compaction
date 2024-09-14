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
		__global__ void kernNaiveScan(int n, int* odata, const int* idata, int d) {
			int index = threadIdx.x + (blockIdx.x * blockDim.x);
            if (index >= n) return;

			if (index >= (1 << d)) {
				odata[index] = idata[index - (1 << d)] + idata[index];
			}
			else {
				odata[index] = idata[index];
			}
		}

		__global__ void kernShiftRight(int n, int* odata, const int* idata) {
			int index = threadIdx.x + (blockIdx.x * blockDim.x);
			if (index >= n) return;

			odata[index] = index == 0 ? 0 : idata[index - 1];
		}

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            timer().startGpuTimer();
            // TODO
			int blockSize = 128;

			int* dev_idata;
			int* dev_odata;
			
			cudaMalloc((void**)&dev_idata, n * sizeof(int));
			checkCUDAError("cudaMalloc dev_idata failed!");
			cudaMalloc((void**)&dev_odata, n * sizeof(int));
			checkCUDAError("cudaMalloc dev_odata failed!");

			cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);
			checkCUDAError("cudaMemcpy idata to dev_idata failed!");

			dim3 fullBlocksPerGrid((n + blockSize - 1) / blockSize);
			
			for (int d = 0; d < ilog2ceil(n); d++) {
				kernNaiveScan << <fullBlocksPerGrid, blockSize >> > (n, dev_odata, dev_idata, d);
				checkCUDAError("kernNaiveScan failed!");
				cudaMemcpy(dev_idata, dev_odata, n * sizeof(int), cudaMemcpyDeviceToDevice);
			}

			kernShiftRight << <fullBlocksPerGrid, blockSize >> > (n, dev_odata, dev_idata);
			cudaMemcpy(odata, dev_odata, n * sizeof(int), cudaMemcpyDeviceToHost);
			checkCUDAError("cudaMemcpy dev_odata to odata failed!");


			
			cudaFree(dev_idata);
			cudaFree(dev_odata);

            timer().endGpuTimer();
        }
    }
}

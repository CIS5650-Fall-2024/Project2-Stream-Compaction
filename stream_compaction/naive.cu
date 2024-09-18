#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"

namespace StreamCompaction {
    namespace Naive {

		const int blockSize = 128;

        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

		__global__ void kernScan(int n, int d, int* odata, const int* idata) {
			int index = threadIdx.x + (blockIdx.x * blockDim.x);
            if (index >= n) {
				return;
			}
            // Use bit shift to calculate 2^ d-1
			if (index >= (1 << (d - 1))) {
				odata[index] = idata[index] + idata[index - (1 << (d - 1))];
			}
			else {
				odata[index] = idata[index];
			}
		}

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
			// Allocate memory on device
			int* dev_idata;
			int* dev_odata;
			cudaMalloc((void**)&dev_idata, n * sizeof(int));
			checkCUDAError("cudaMalloc dev_idata failed!");
			cudaMalloc((void**)&dev_odata, n * sizeof(int));
			checkCUDAError("cudaMalloc dev_odata failed!");

			// Copy data from host to device
			cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);
			checkCUDAError("cudaMemcpy idata to dev_idata failed!");

			dim3 blocksPerGrid((n + blockSize - 1) / blockSize);

            timer().startGpuTimer();
			for (int d = 1; d <= ilog2ceil(n); d++) {
				kernScan <<<blocksPerGrid, blockSize>>> (n, d, dev_odata, dev_idata);
				std::swap(dev_odata, dev_idata);
			}
			std::swap(dev_odata, dev_idata);
			odata[0] = 0;

            timer().endGpuTimer();

			// Copy data from device to host while shifting right for exclusive scan
			cudaMemcpy(odata + 1, dev_odata, (n - 1) * sizeof(int), cudaMemcpyDeviceToHost);
			checkCUDAError("cudaMemcpy dev_odata to odata failed!");

			cudaFree(dev_idata);
			cudaFree(dev_odata);
        }
    }
}

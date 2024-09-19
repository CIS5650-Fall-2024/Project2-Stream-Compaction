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

        __global__ void kernNaiveScan(int n, int offset, int* odata, int* idata) {

			int index = threadIdx.x + (blockIdx.x * blockDim.x);

            if (index >= n) {
                return;
            }

			if (index >= offset) {
				odata[index] = idata[index - offset] + idata[index];
			}
			else {
				odata[index] = idata[index];
			}
			
        }
        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            // TODO
            // use ilog2ceil(n) 
			dim3 blocksPerGrid = (n + blockSize - 1) / blockSize;

			// ping-pong buffers
			int* dev_data1;
            int* dev_data2;

            cudaMalloc((void**)&dev_data1, n * sizeof(int));
            checkCUDAError("cudaMalloc dev_data1 failed!");

			cudaMalloc((void**)&dev_data2, n * sizeof(int));
			checkCUDAError("cudaMalloc dev_data2 failed!");

			cudaMemcpy(dev_data1, idata, n * sizeof(int), cudaMemcpyHostToDevice);
			checkCUDAError("cudaMemcpy idata to dev_data1 failed!");

            // Start Timer
			timer().startGpuTimer();

			for (int d = 1; d <= ilog2ceil(n); d++) {
				int offset = 1 << (d - 1);
				kernNaiveScan << <blocksPerGrid, blockSize >> > (n, offset, dev_data2, dev_data1);
                checkCUDAError("kernNaiveScan failed!");
				std::swap(dev_data1, dev_data2);
			}
			// End Timer
            timer().endGpuTimer();

			// From inclusive to exclusive, shift right by 1, fill the first element with 0
            odata[0] = 0;
            // Real data stored in dev_data1
            cudaMemcpy(odata + 1, dev_data1, sizeof(int) * (n - 1), cudaMemcpyDeviceToHost);
            checkCUDAError("cudaMemcpy dev_data1 to odata failed!");

            cudaFree(dev_data1);
			cudaFree(dev_data2);
        }
    }
}

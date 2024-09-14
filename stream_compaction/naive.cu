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
		int getLog2(int n) {
			int log2 = 0;
			while (n >>= 1) {
				log2++;
			}
			return log2+1;
		}

		__global__ void kernScan(int n, int* odata, const int* idata, int log2_n) {
			int index = threadIdx.x + (blockIdx.x * blockDim.x);
			if (index >= n) {
				return;
			}
			// exclusive scan
			odata[index] = (index > 0) ? idata[index - 1] : 0;
			__syncthreads();

			for (int d = 1; d <= log2_n; ++d) {
				int t = 1 << (d - 1);
				int temp = 0;
				if (index >= t) temp = odata[index - t];  // Load the previous step's value
				__syncthreads();  // Synchronize before updating
				if (index >= t) odata[index] += temp;  // Update the current value
				__syncthreads();  // Synchronize after updating
			}
		}

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            timer().startGpuTimer();
            // TODO
			int log2_n = getLog2(n);
			//printf("n: %d\n", n);
			//printf("log2_n: %d\n", log2_n);
			// call kernel
			int* dev_idata;
			int* dev_odata;
			cudaMalloc((void**)&dev_idata, n * sizeof(int));
			cudaMalloc((void**)&dev_odata, n * sizeof(int));
			cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);
			kernScan << <1, n >> > (n, dev_odata, dev_idata, log2_n);
			cudaMemcpy(odata, dev_odata, n * sizeof(int), cudaMemcpyDeviceToHost);
			cudaFree(dev_idata);
			cudaFree(dev_odata);


            timer().endGpuTimer();
        }
    }
}

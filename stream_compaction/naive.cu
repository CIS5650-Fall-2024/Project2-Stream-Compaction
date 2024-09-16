#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"

#define blockSize 512

namespace StreamCompaction {
	namespace Naive {
		using StreamCompaction::Common::PerformanceTimer;
		PerformanceTimer& timer()
		{
			static PerformanceTimer timer;
			return timer;
		}
		// TODO: __global__
		// for every element do a scan in logn
		__global__ void kernScanNaive(int n, int d, int* odata, const int* idata) {
			int index = (blockIdx.x * blockDim.x) + threadIdx.x;
			if (index >= n) return;

			if (index >= 1 << (d - 1))
				odata[index] = idata[index - (1 << (d - 1))] + idata[index];
			else odata[index] = idata[index];
		}

		/**
		 * Performs prefix-sum (aka scan) on idata, storing the result into odata.
		 */
		void scan(int n, int* odata, const int* idata) {
			timer().startGpuTimer();
			//copy to device
			int* dev_idata;
			int* dev_odata;
			cudaMalloc((void**)&dev_idata, n * sizeof(int));
			cudaMalloc((void**)&dev_odata, n * sizeof(int));

			cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);

			dim3 blocksPerGrid((n + blockSize - 1) / blockSize);
			for (int d = 1; d <= ilog2ceil(n); d++)
			{
				kernScanNaive << < blocksPerGrid, blockSize >> > (n, d, dev_odata, dev_idata);
				std::swap(dev_odata, dev_idata);
			}

			odata[0] = 0;
			cudaMemcpy(odata + 1, dev_idata, (n - 1) * sizeof(int), cudaMemcpyDeviceToHost);
			cudaFree(dev_idata);
			cudaFree(dev_odata);
			timer().endGpuTimer();
		}
	}
}

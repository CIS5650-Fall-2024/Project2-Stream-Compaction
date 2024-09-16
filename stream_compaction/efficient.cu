#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

#define blockSize 512

namespace StreamCompaction {
	namespace Efficient {
		using StreamCompaction::Common::PerformanceTimer;
		PerformanceTimer& timer()
		{
			static PerformanceTimer timer;
			return timer;
		}

		__global__ void kernUpSweep(int n, int d, int* odata)
		{
			int index = (blockIdx.x * blockDim.x) + threadIdx.x;
			if (index >= n) return;

			int k = index * (1 << (d + 1));

			odata[k + (1 << (d + 1)) - 1] += odata[k + (1 << d) - 1];
		}

		__global__ void kernDownSweep(int n, int d, int* odata)
		{
			int index = (blockIdx.x * blockDim.x) + threadIdx.x;
			if (index >= n) return;

			int k = index * (1 << (d + 1));

			int t = odata[k + (1 << d) - 1];
			odata[k + (1 << d) - 1] = odata[k + (1 << (d + 1)) - 1];
			odata[k + (1 << (d + 1)) - 1] += t;
		}

		/**
		 * Performs prefix-sum (aka scan) on idata, storing the result into odata.
		 */
		void scan(int n, int* odata, const int* idata) {
			timer().startGpuTimer();

			int* dev_indices;
			cudaMalloc((void**)&dev_indices, n * sizeof(int));
			cudaMemcpy(dev_indices, idata, n * sizeof(int), cudaMemcpyHostToDevice);

			//upsweep
			dim3 blocksPerGrid((n + blockSize - 1) / blockSize);
			for (int d = 0; d < ilog2ceil(n); d++)
			{
				int kSteps = n >> (d + 1);
				dim3 blocksPerGrid((kSteps + blockSize - 1) / blockSize);
				kernUpSweep << < blocksPerGrid, blockSize >> > (kSteps, d, dev_indices);
			}

			//downsweep
			cudaMemcpy(odata, dev_indices, n * sizeof(int), cudaMemcpyDeviceToHost);
			odata[n - 1] = 0;
			cudaMemcpy(dev_indices, odata, n * sizeof(int), cudaMemcpyHostToDevice);

			for (int d = ilog2ceil(n) - 1; d >= 0; d--)
			{
				int kSteps = n >> (d + 1);
				// non-power-of-2
				if (kSteps << (d + 1) != n)  kSteps = n >> d;
				dim3 blocksPerGrid((kSteps + blockSize - 1) / blockSize);
				kernDownSweep << < blocksPerGrid, blockSize >> > (kSteps, d, dev_indices);
			}

			cudaMemcpy(odata, dev_indices, n * sizeof(int), cudaMemcpyDeviceToHost);
			cudaFree(dev_indices);
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
		int compact(int n, int* odata, const int* idata) {
			timer().startGpuTimer();
			//a buffer used to r/w single element of device ptr
			int* buffer = new int[n];

			int* dev_idata;
			int* dev_label;
			cudaMalloc((void**)&dev_idata, n * sizeof(int));
			cudaMalloc((void**)&dev_label, n * sizeof(int));
			cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);

			//get labels
			dim3 blocksPerGrid((n + blockSize - 1) / blockSize);
			Common::kernMapToBoolean << <blocksPerGrid, blockSize >> > (n, dev_label, dev_idata);

			//scan begin
			int* dev_indices;
			cudaMalloc((void**)&dev_indices, n * sizeof(int));
			cudaMemcpy(dev_indices, dev_label, n * sizeof(int), cudaMemcpyDeviceToDevice);

			//upsweep
			for (int d = 0; d < ilog2ceil(n); d++)
			{
				int kSteps = n >> (d + 1);
				dim3 blocksPerGrid((kSteps + blockSize - 1) / blockSize);
				kernUpSweep << < blocksPerGrid, blockSize >> > (kSteps, d, dev_indices);
			}

			//downsweep
			cudaMemcpy(buffer, dev_indices, n * sizeof(int), cudaMemcpyDeviceToHost);
			buffer[n - 1] = 0;
			cudaMemcpy(dev_indices, buffer, n * sizeof(int), cudaMemcpyHostToDevice);

			for (int d = ilog2ceil(n) - 1; d >= 0; d--)
			{
				int kSteps = n >> (d + 1);
				// non-power-of-2
				if (kSteps << (d + 1) != n)  kSteps = n >> d;
				dim3 blocksPerGrid((kSteps + blockSize - 1) / blockSize);
				kernDownSweep << < blocksPerGrid, blockSize >> > (kSteps, d, dev_indices);
			}

			//calculate total numbers
			cudaMemcpy(buffer, dev_indices, n * sizeof(int), cudaMemcpyDeviceToHost);
			int count = buffer[n - 1];
			cudaMemcpy(buffer, dev_label, n * sizeof(int), cudaMemcpyDeviceToHost);
			count += buffer[n - 1];

			//scatter to labeled data
			int* dev_odata;
			cudaMalloc((void**)&dev_odata, count * sizeof(int));
			Common::kernScatter << <blocksPerGrid, blockSize >> > (n, dev_odata,
				dev_idata, dev_label, dev_indices);

			//copy to cpu output
			cudaMemcpy(odata, dev_odata, count * sizeof(int), cudaMemcpyDeviceToHost);

			delete[] buffer;
			cudaFree(dev_idata);
			cudaFree(dev_odata);
			cudaFree(dev_label);
			cudaFree(dev_indices);
			timer().endGpuTimer();
			return count;
		}
	}
}

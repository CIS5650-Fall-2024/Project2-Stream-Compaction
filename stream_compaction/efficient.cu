#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

#include <thrust/device_ptr.h>

#define blockSize 128

namespace StreamCompaction {
	namespace Efficient {
		using StreamCompaction::Common::PerformanceTimer;
		PerformanceTimer& timer() {
			static PerformanceTimer timer;
			return timer;
		}

		__global__ void kernUpSweep(int n, int d, int* odata) {
			int index = (blockIdx.x * blockDim.x) + threadIdx.x;
			if (index >= n) return;

			int k = index * (1 << (d + 1));

			odata[k + (1 << (d + 1)) - 1] += odata[k + (1 << d) - 1];
		}

		__global__ void kernDownSweep(int n, int d, int* odata) {
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

			int N = 1 << ilog2ceil(n);
			int* dev_indices;
			//cudaMalloc((void**)&dev_indices, n * sizeof(int));
			cudaMalloc((void**)&dev_indices, N * sizeof(int));
			cudaMemcpy(dev_indices, idata, n * sizeof(int), cudaMemcpyHostToDevice);
			thrust::device_ptr<int> thrust_indices(dev_indices);

			timer().startGpuTimer();

			//upsweep
			for (int d = 0; d < ilog2ceil(N); d++){
				int kSteps = N >> (d + 1);
				dim3 blocksPerGrid((kSteps + blockSize - 1) / blockSize);
				kernUpSweep << < blocksPerGrid, blockSize >> > (kSteps, d, dev_indices);
			}

			//downsweep
			thrust_indices[N - 1] = 0;

			for (int d = ilog2ceil(N) - 1; d >= 0; d--){
				int kSteps = N >> (d + 1);
				dim3 blocksPerGrid((kSteps + blockSize - 1) / blockSize);
				kernDownSweep << < blocksPerGrid, blockSize >> > (kSteps, d, dev_indices);
			}

			timer().endGpuTimer();
			cudaMemcpy(odata, dev_indices, n * sizeof(int), cudaMemcpyDeviceToHost);
			cudaFree(dev_indices);
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

			int N = 1 << ilog2ceil(n);
			int* dev_idata;
			int* dev_label;
			cudaMalloc((void**)&dev_idata, n * sizeof(int));
			cudaMalloc((void**)&dev_label, n * sizeof(int));
			cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);
			cudaMemcpy(dev_label, dev_idata, n * sizeof(int), cudaMemcpyDeviceToDevice);
			thrust::device_ptr<int> thrust_label(dev_label);

			int* dev_indices;
			cudaMalloc((void**)&dev_indices, N * sizeof(int));
			thrust::device_ptr<int> thrust_indices(dev_indices);

			timer().startGpuTimer();

			//get labels
			dim3 blocksPerGrid((n + blockSize - 1) / blockSize);
			Common::kernMapToBoolean << <blocksPerGrid, blockSize >> > (n, dev_label, dev_idata);

			//scan begin
			cudaMemcpy(dev_indices, dev_label, n * sizeof(int), cudaMemcpyDeviceToDevice);

			//upsweep
			for (int d = 0; d < ilog2ceil(N); d++){
				int kSteps = N >> (d + 1);
				dim3 blocksPerGrid((kSteps + blockSize - 1) / blockSize);
				kernUpSweep << < blocksPerGrid, blockSize >> > (kSteps, d, dev_indices);
			}

			//downsweep
			thrust_indices[N - 1] = 0;

			for (int d = ilog2ceil(N) - 1; d >= 0; d--){
				int kSteps = N >> (d + 1);
				dim3 blocksPerGrid((kSteps + blockSize - 1) / blockSize);
				kernDownSweep << < blocksPerGrid, blockSize >> > (kSteps, d, dev_indices);
			}

			//calculate total numbers
			int count = thrust_indices[n - 1] + thrust_label[n - 1];

			//scatter to labeled data
			int* dev_odata;
			cudaMalloc((void**)&dev_odata, count * sizeof(int));
			Common::kernScatter << <blocksPerGrid, blockSize >> > (n, dev_odata,
				dev_idata, dev_label, dev_indices);

			timer().endGpuTimer();

			//copy to cpu output
			cudaMemcpy(odata, dev_odata, count * sizeof(int), cudaMemcpyDeviceToHost);
			cudaFree(dev_idata);
			cudaFree(dev_odata);
			cudaFree(dev_label);
			cudaFree(dev_indices);
			return count;
		}
	}
}

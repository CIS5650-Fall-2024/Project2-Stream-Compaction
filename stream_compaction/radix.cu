#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "radix.h"
#include "efficient.h"

#include <thrust/device_ptr.h>

#define blockSize 128

namespace StreamCompaction {
	namespace Radix {
		using StreamCompaction::Common::PerformanceTimer;
		PerformanceTimer& timer()
		{
			static PerformanceTimer timer;
			return timer;
		}

		__global__ void kernSplit(int n, int bit, const int* i, int* b, int* e) {
			int index = (blockIdx.x * blockDim.x) + threadIdx.x;
			if (index >= n) return;

			b[index] = (i[index] >> bit) & 1;
			e[index] = b[index] ? 0 : 1;
		}

		__global__ void kernAddress(int n, int count, const int* b, const int* f, int* d) {
			int index = (blockIdx.x * blockDim.x) + threadIdx.x;
			if (index >= n) return;

			//d[i] = b[i] ? t[i] : f[i]
			//d[i] = b[i] ? i – f[i] + count : f[i]
			d[index] = b[index] ? index - f[index] + count : f[index];
		}

		__global__ void kernScatter(int n, const int* i, const int* d, int* o) {
			int index = (blockIdx.x * blockDim.x) + threadIdx.x;
			if (index >= n) return;

			o[d[index]] = i[index];
		}


		void sort(int n, int* odata, const int* idata) {
			int N = 1 << ilog2ceil(n);

			int* dev_i;
			int* dev_b;
			int* dev_e;
			int* dev_d;
			int* dev_f;
			int* dev_o;

			cudaMalloc((void**)&dev_i, n * sizeof(int));
			cudaMalloc((void**)&dev_b, n * sizeof(int));
			cudaMalloc((void**)&dev_e, n * sizeof(int));
			cudaMalloc((void**)&dev_d, n * sizeof(int));
			cudaMalloc((void**)&dev_f, N * sizeof(int));
			cudaMalloc((void**)&dev_o, n * sizeof(int));

			cudaMemcpy(dev_i, idata, n * sizeof(int), cudaMemcpyHostToDevice);

			dim3 blocksPerGrid((n + blockSize - 1) / blockSize);
			timer().startGpuTimer();
			for (int bit = 0; bit < 31; bit++) {
				//Implement spilt
				kernSplit << <blocksPerGrid, blockSize >> > (n, bit, dev_i, dev_b, dev_e);

				//scan begin
				cudaMemcpy(dev_f, dev_e, n * sizeof(int), cudaMemcpyDeviceToDevice);

				//upsweep
				for (int d = 0; d < ilog2ceil(N); d++) {
					int kSteps = N >> (d + 1);
					dim3 blocksPerGrid((kSteps + blockSize - 1) / blockSize);
					Efficient::kernUpSweep << < blocksPerGrid, blockSize >> > (kSteps, d, dev_f);
				}

				//downsweep
				thrust::device_ptr<int> thrust_f(dev_f);
				thrust_f[N - 1] = 0;

				for (int d = ilog2ceil(N) - 1; d >= 0; d--) {
					int kSteps = 0;
					kSteps = N >> (d + 1);
					dim3 blocksPerGrid((kSteps + blockSize - 1) / blockSize);
					Efficient::kernDownSweep << < blocksPerGrid, blockSize >> > (kSteps, d, dev_f);
				}

				//calculate total numbers
				thrust::device_ptr<int> thrust_e(dev_e);
				int count = thrust_f[n - 1] + thrust_e[n - 1];

				//Get d
				kernAddress << <blocksPerGrid, blockSize >> > (n, count, dev_b, dev_f, dev_d);

				//Implement scatter
				kernScatter << <blocksPerGrid, blockSize >> > (n, dev_i, dev_d, dev_o);

				std::swap(dev_o, dev_i);
			}

			timer().endGpuTimer();
			cudaMemcpy(odata, dev_i, n * sizeof(int), cudaMemcpyDeviceToHost);

			cudaFree(dev_i);
			cudaFree(dev_b);
			cudaFree(dev_e);
			cudaFree(dev_d);
			cudaFree(dev_o);
			cudaFree(dev_f);
		}
	}
}


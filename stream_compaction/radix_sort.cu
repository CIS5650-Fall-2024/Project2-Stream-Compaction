#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

namespace StreamCompaction {
	namespace RadixSort {
		using StreamCompaction::Common::PerformanceTimer;
		PerformanceTimer& timer()
		{
			static PerformanceTimer timer;
			return timer;
		}

		__global__ void kernInitEF(int n, int *e, int *f, const int *idata, int bit) {
			int index = (blockIdx.x * blockDim.x) + threadIdx.x;
			if (index < n) {
				e[index] = (idata[index] & (1 << bit)) ? 0 : 1;
				f[index] = e[index];
			}
		}

		__global__ void kernUpSweep(int n, int *data, int step) {
			int index = (blockIdx.x * blockDim.x) + threadIdx.x;
			if (index < n && index % step == 0) {
				data[index + step - 1] += data[index + (step >> 1) - 1];
			}
		}

		__global__ void kernSetLastZero(int n, int *data) {
			int index = (blockIdx.x * blockDim.x) + threadIdx.x;
			if (index == n - 1) {
				data[index] = 0;
			}
		}

		__global__ void kernDownSweep(int n, int *data, int step) {
			int index = (blockIdx.x * blockDim.x) + threadIdx.x;
			if (index < n && index % step == 0) {
				int idx1 = index + (step >> 1) - 1, idx2 = index + step - 1;
				int t = data[idx1];
				data[idx1] = data[idx2];
				data[idx2] += t;
			}
		}

		__global__ void kernTotalFalses(int n, int *count, const int *e, const int *f) {
			int index = (blockIdx.x * blockDim.x) + threadIdx.x;
			if (index == n - 1) {
				count[0] = e[index] + f[index];
			}
		}

		__global__ void kernComputeT(int n, int *t, const int *f, const int *count) {
			int index = (blockIdx.x * blockDim.x) + threadIdx.x;
			if (index < n) {
				t[index] = index - f[index] + count[0];
			}
		}

		__global__ void kernScatterOut(int n, int *odata, const int *idata, const int *e, const int *f, const int *t) {
			int index = (blockIdx.x * blockDim.x) + threadIdx.x;
			if (index < n) {
				int d = e[index] ? f[index] : t[index];
				odata[d] = idata[index];
			}
		}

		void sort(int n, int *odata, const int *idata) {
			int *dev_odata, *dev_idata, *dev_e, *dev_f, *dev_t, *dev_count;
			int rounds = ilog2ceil(n);
			int size = 1 << rounds;
			cudaMalloc((void**)&dev_odata, n * sizeof(int));
			checkCUDAError("cudaMalloc dev_odata failed!");
			cudaMalloc((void**)&dev_idata, n * sizeof(int));
			checkCUDAError("cudaMalloc dev_idata failed!");
			cudaMalloc((void**)&dev_e, n * sizeof(int));
			checkCUDAError("cudaMalloc dev_e failed!");
			cudaMalloc((void**)&dev_f, size * sizeof(int));
			checkCUDAError("cudaMalloc dev_f failed!");
			cudaMalloc((void**)&dev_t, n * sizeof(int));
			checkCUDAError("cudaMalloc dev_t failed!");
			cudaMalloc((void**)&dev_count, sizeof(int));
			checkCUDAError("cudaMalloc dev_count failed!");
			cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);
			dim3 blocks((n + blockSize - 1) / blockSize);

			timer().startGpuTimer();
			for (int bit = 0; bit < 31; bit++) {
				kernInitEF<<<blocks, blockSize>>>(n, dev_e, dev_f, dev_idata, bit);
				checkCUDAError("kernInitEF failed!");
				int step = 2;
				for (int i = 0; i < rounds; i++) {
					kernUpSweep<<<blocks, blockSize>>>(size, dev_f, step);
					checkCUDAError("kernUpSweep failed!");
					step <<= 1;
				}
				kernSetLastZero<<<blocks, blockSize>>>(size, dev_f);
				checkCUDAError("kernSetLastZero failed!");
				step = size;
				for (int i = 0; i < rounds; i++) {
					kernDownSweep<<<blocks, blockSize>>>(size, dev_f, step);
					checkCUDAError("kernDownSweep failed!");
					step >>= 1;
				}
				kernTotalFalses<<<blocks, blockSize>>>(n, dev_count, dev_e, dev_f);
				checkCUDAError("kernTotalFalses failed!");
				kernComputeT<<<blocks, blockSize>>>(n, dev_t, dev_f, dev_count);
				checkCUDAError("kernComputeT failed!");
				kernScatterOut<<<blocks, blockSize>>>(n, dev_odata, dev_idata, dev_e, dev_f, dev_t);
				checkCUDAError("kernScatterOut failed!");
				std::swap(dev_odata, dev_idata);
			}
			timer().endGpuTimer();
			cudaMemcpy(odata, dev_idata, n * sizeof(int), cudaMemcpyDeviceToHost);
			cudaFree(dev_odata);
			cudaFree(dev_idata);
			cudaFree(dev_e);
			cudaFree(dev_f);
			cudaFree(dev_t);
			cudaFree(dev_count);
		}
	}
}

#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

#define blockSize 128

namespace StreamCompaction {
    namespace Efficient {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

		__global__ void kernUpSweep(int n, int d, int* odata) {
			int index = threadIdx.x + (blockIdx.x * blockDim.x);
			if (index >= n) { return; }
			// Use bit shift instead of modulus to check if index is divisible by 2^(d+1)
			// The - 1 creates a bit mask that makes the last d+1 bits 1
			// With the & operation, we can check if the last d+1 bits are all 0
			if (!(index & ((1 << (d + 1)) - 1))) {
				odata[index + (1 << (d + 1)) - 1] += odata[index + (1 << d) - 1];
			}
		}

        __global__ void kernDownSweep(int n, int d, int* odata) {
			int index = threadIdx.x + (blockIdx.x * blockDim.x);
            if (index >= n) { return; }
			int offset = 1 << (d + 1);
            if (!(index & (offset - 1))) {
				int t = odata[index + (1 << d) - 1];
				odata[index + (1 << d) - 1] = odata[index + offset - 1];
				odata[index + offset - 1] += t;
            }
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {

			int d = ilog2ceil(n);
			int size = 1 << d;
			dim3 blocksPerGrid((size + blockSize - 1) / blockSize);

			int* dev_data;
			cudaMalloc((void**)&dev_data, size * sizeof(int));
			checkCUDAError("cudaMalloc dev_data failed!");
			cudaMemcpy(dev_data, idata, n * sizeof(int), cudaMemcpyHostToDevice);

            timer().startGpuTimer();
            
            // Upsweep
			for (int i = 0; i < d - 1; i++) {
				kernUpSweep <<<blocksPerGrid, blockSize >>> (size, i, dev_data);
				checkCUDAError("kernUpSweep failed!");
            }

			cudaMemset(dev_data + size - 1, 0, sizeof(int));

			// Downsweep
            for (int i = d - 1; i >= 0; i--) {
				kernDownSweep <<<blocksPerGrid, blockSize >>> (size, i, dev_data);
            }

            timer().endGpuTimer();

			cudaMemcpy(odata, dev_data, n * sizeof(int), cudaMemcpyDeviceToHost);
			checkCUDAError("cudaMemcpy dev_data to odata failed!");
			cudaFree(dev_data);
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
        int compact(int n, int *odata, const int *idata) {

            
			int* dev_bools;
			int* dev_indices;
			int* dev_idata;
			int* dev_odata;

			int d = ilog2ceil(n);
			int size = 1 << d;
			dim3 blocksPerGrid((size + blockSize - 1) / blockSize);
            
			cudaMalloc((void**)&dev_bools, size * sizeof(int));
			checkCUDAError("cudaMalloc dev_bools failed!");
			cudaMalloc((void**)&dev_indices, size * sizeof(int));
			checkCUDAError("cudaMalloc dev_indices failed!");
			cudaMalloc((void**)&dev_idata, size * sizeof(int));
			checkCUDAError("cudaMalloc dev_idata failed!");
			cudaMalloc((void**)&dev_odata, size * sizeof(int));
			checkCUDAError("cudaMalloc dev_odata failed!");
			cudaMemcpy(dev_idata, idata, size * sizeof(int), cudaMemcpyHostToDevice);

			timer().startGpuTimer();

			// Map to boolean
			StreamCompaction::Common::kernMapToBoolean << <blocksPerGrid, blockSize >> > (n, dev_bools, dev_idata);
			checkCUDAError("kernMapToBoolean failed!");

			// Scan
			cudaMemcpy(dev_indices, dev_bools, size * sizeof(int), cudaMemcpyDeviceToDevice);
			checkCUDAError("cudaMemcpy dev_bools to dev_indices failed!");

			// Upsweep
			for (int i = 0; i < d - 1; i++) {
				kernUpSweep << <blocksPerGrid, blockSize >> > (size, i, dev_indices);
				checkCUDAError("kernUpSweep failed!");
			}

			cudaMemset(dev_indices + size - 1, 0, sizeof(int));
			checkCUDAError("cudaMemset dev_indices failed!");

			// Downsweep
			for (int i = d - 1; i >= 0; i--) {
				kernDownSweep << <blocksPerGrid, blockSize >> > (size, i, dev_indices);
				checkCUDAError("kernDownSweep failed!");
			}
			StreamCompaction::Common::kernScatter << <blocksPerGrid, blockSize >> > (size, dev_odata, dev_idata, dev_bools, dev_indices);
			checkCUDAError("kernScatter failed!");


            timer().endGpuTimer();
            
			int count = 0;
			cudaMemcpy(&count, dev_indices + size - 1, sizeof(int), cudaMemcpyDeviceToHost);
			checkCUDAError("cudaMemcpy dev_indices to count failed!");

			cudaMemcpy(odata, dev_odata, count * sizeof(int), cudaMemcpyDeviceToHost);
			checkCUDAError("cudaMemcpy dev_odata to odata failed!");

			cudaFree(dev_bools);
			cudaFree(dev_indices);
			cudaFree(dev_idata);
			cudaFree(dev_odata);

			return count;
        }
    }
}

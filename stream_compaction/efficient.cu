#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

namespace StreamCompaction {
    namespace Efficient {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        __global__ void kernUpSweep(int n, int d, int* data) {
            int index = threadIdx.x + (blockIdx.x * blockDim.x);
			if (index >= n) {
				return;
			}
#if ThreadReduction == 0
			int d1 = 1 << d;
			int d2 = 1 << (d + 1);

			if (index % d2 == 0) {
				data[index + d2 - 1] += data[index + d1 - 1];
			}
#else		
			data[(index + 1) * (1 << (d + 1)) - 1] += data[index * (1 << (d + 1)) + (1 << d) - 1 ];
#endif
        }

		__global__ void kernDownSweep(int n, int d, int* data) {
            int index = threadIdx.x + (blockIdx.x * blockDim.x);
			if (index >= n) {
				return;
			}
#if ThreadReduction == 0
			int d1 = 1 << d;
			int d2 = 1 << (d + 1);

			if (index % d2 == 0) {
				int t = data[index + d1 - 1];
				data[index + d1 - 1] = data[index + d2 - 1];
				data[index + d2 - 1] += t;
			}
#else		
			int stride1 = (index + 1) * (1 << (d + 1)) - 1;
			int stride2 = index * (1 << (d + 1)) + (1 << d) - 1;
			int temp = data[stride2];
			data[stride2] = data[stride1];
			data[stride1] += temp;
#endif
		}

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {

			int* dev_data;
			int n_powerOf2 = 1 << ilog2ceil(n);

			cudaMalloc((void**)&dev_data, n_powerOf2 * sizeof(int));
			checkCUDAError("cudaMalloc dev_data failed!");

			cudaMemcpy(dev_data, idata, n * sizeof(int), cudaMemcpyHostToDevice);
			checkCUDAError("cudaMemcpy idata to dev_data failed!");
			cudaMemset(dev_data + n, 0, (n_powerOf2 - n) * sizeof(int));

			dim3 blocksPerGrid = (n_powerOf2 + blockSize - 1) / blockSize;

            timer().startGpuTimer();

			// Up-Sweep
			for (int d = 0; d <= ilog2ceil(n_powerOf2) - 1; d++) {
#if ThreadReduction == 1
				int numberOfThreads = n_powerOf2 / (1 << (d + 1));
				kernUpSweep << <blocksPerGrid, blockSize >> > (numberOfThreads, d, dev_data);
#else
				kernUpSweep << <blocksPerGrid, blockSize >> > (n_powerOf2, d, dev_data);
#endif
				checkCUDAError("kernUpSweep failed!");
			}

			// Set root to 0
			cudaMemset(dev_data + n_powerOf2 - 1, 0, sizeof(int));

			// Down-Sweep
			for (int d = ilog2ceil(n_powerOf2) - 1; d >= 0; d--) {
#if ThreadReduction == 1
				int numberOfThreads = n_powerOf2 / (1 << (d + 1));
				kernDownSweep << <blocksPerGrid, blockSize >> > (numberOfThreads, d, dev_data);
#else
				kernDownSweep << <blocksPerGrid, blockSize >> > (n_powerOf2, d, dev_data);
#endif
				checkCUDAError("kernDownSweep failed!");
			}

            timer().endGpuTimer();

			// Copy result to odata
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
			int* dev_idata;
			int* dev_odata;
			int* dev_temp;
			int* dev_scanArr;

			int n_powerOf2 = 1 << ilog2ceil(n);

			cudaMalloc((void**)&dev_idata, n_powerOf2 * sizeof(int));
			checkCUDAError("cudaMalloc dev_idata failed!");
			cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);
			checkCUDAError("cudaMemcpy idata to dev_idata failed!");
			cudaMemset(dev_idata + n, 0, (n_powerOf2 - n) * sizeof(int));

			cudaMalloc((void**)&dev_odata, n_powerOf2 * sizeof(int));
			checkCUDAError("cudaMalloc dev_odata failed!");

			cudaMalloc((void**)&dev_temp, n_powerOf2 * sizeof(int));
			checkCUDAError("cudaMalloc temp failed!");

			cudaMalloc((void**)&dev_scanArr, n_powerOf2 * sizeof(int));
			checkCUDAError("cudaMalloc scanArr failed!");

			dim3 blocksPerGrid = (n_powerOf2 + blockSize - 1) / blockSize;

			timer().startGpuTimer();
			// Step 1 Compute temporary array containing
			StreamCompaction::Common::kernMapToBoolean << <blocksPerGrid, blockSize >> > (n_powerOf2, dev_temp, dev_idata);
			checkCUDAError("kernMapToBoolean failed!");

			cudaMemcpy(dev_scanArr, dev_temp, n * sizeof(int), cudaMemcpyDeviceToDevice);
			// Step 2 Run exclusive scan on the temp array

			// Up-Sweep
			for (int d = 0; d <= ilog2ceil(n_powerOf2) - 1; d++) {
#if ThreadReduction == 1
				int numberOfThreads = n_powerOf2 / (1 << (d + 1));
				kernUpSweep << <blocksPerGrid, blockSize >> > (numberOfThreads, d, dev_scanArr);
#else
				kernUpSweep << <blocksPerGrid, blockSize >> > (n_powerOf2, d, dev_scanArr);
#endif
				checkCUDAError("kernUpSweep failed!");
			}

			// Set root to 0
			cudaMemset(dev_scanArr + n_powerOf2 - 1, 0, sizeof(int));

			// Down-Sweep
			for (int d = ilog2ceil(n_powerOf2) - 1; d >= 0; d--) {
#if ThreadReduction == 1
				int numberOfThreads = n_powerOf2 / (1 << (d + 1));
				kernDownSweep << <blocksPerGrid, blockSize >> > (numberOfThreads, d, dev_scanArr);
#else
				kernDownSweep << <blocksPerGrid, blockSize >> > (n_powerOf2, d, dev_scanArr);
#endif
				checkCUDAError("kernDownSweep failed!");
			}

			// Step 3 Scatter
			Common::kernScatter << <blocksPerGrid, blockSize >> > (n, dev_odata, dev_idata, dev_temp, dev_scanArr);
            timer().endGpuTimer();

			// Copy result to odata
			cudaMemcpy(odata, dev_odata, n * sizeof(int), cudaMemcpyDeviceToHost);
			checkCUDAError("cudaMemcpy dev_odata to odata failed!");

			int cnt = 0;
			cudaMemcpy(&cnt, dev_scanArr + n_powerOf2 - 1, sizeof(int), cudaMemcpyDeviceToHost);
			checkCUDAError("cudaMemcpy dev_scanArr to cnt failed!");

			cudaFree(dev_odata);
			cudaFree(dev_idata);
			cudaFree(dev_temp);
			cudaFree(dev_scanArr);

            return cnt;
        }
    }
}

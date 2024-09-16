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

		__global__ void reduction(int n, int* idata, int d) {
			int index = threadIdx.x + (blockIdx.x * blockDim.x);
			if (index >= n || index % (1 << (d + 1)) != 0) return;

			idata[index + (1 << (d + 1)) - 1] += idata[index + (1 << d) - 1];
		}

		__global__ void kernUpSweep(int n, int* odata, int d) {
			int index = threadIdx.x + (blockIdx.x * blockDim.x);
			if (index >= n || index % (1 << (d + 1)) != 0) return;
            
			odata[index + (1 << (d + 1)) - 1] += odata[index + (1 << d) - 1];
		}

		__global__ void kernDownSweep(int n, int* odata, int d) {
			int index = threadIdx.x + (blockIdx.x * blockDim.x);
			if (index >= n || index % (1 << (d + 1)) != 0) return;


			int t = odata[index + (1 << d) - 1];
			odata[index + (1 << d) - 1] = odata[index + (1 << (d + 1)) - 1];
			odata[index + (1 << (d + 1)) - 1] += t;
		}

		__global__ void computeTempArray(int n, int* odata, const int* idata) {
			int index = threadIdx.x + (blockIdx.x * blockDim.x);
			if (index >= n) return;

			odata[index] = idata[index] == 0 ? 0 : 1;
		}

		__global__ void scatter(int n, int* odata, const int* idata, const int* bools, const int* scan) {	
			int index = threadIdx.x + (blockIdx.x * blockDim.x);
			if (index >= n) return;

			if (idata[index] != 0) {
				odata[scan[index]] = idata[index];
			}
		}

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            //timer().startGpuTimer();
            // TODO
			int blockSize = 128;

			int* dev_odata;

			cudaMalloc((void**)&dev_odata, n * sizeof(int));
			checkCUDAError("cudaMalloc dev_odata failed!");
			
			cudaMemcpy(dev_odata, idata, n * sizeof(int), cudaMemcpyHostToDevice);
			checkCUDAError("cudaMemcpy idata to dev_idata failed!");

			dim3 fullBlocksPerGrid((n + blockSize - 1) / blockSize);

			for (int d = 0; d < ilog2ceil(n); d++) {
				kernUpSweep << <fullBlocksPerGrid, blockSize >> > (n, dev_odata, d);
				checkCUDAError("kernUpSweep failed!");
				cudaDeviceSynchronize();
			}

			cudaMemcpy(odata, dev_odata, n * sizeof(int), cudaMemcpyDeviceToHost);
			checkCUDAError("cudaMemcpy dev_odata to odata failed!");


			cudaMemset(dev_odata + n - 1, 0, sizeof(int));
			for (int d = ilog2ceil(n) - 1; d >= 0; d--) {
				kernDownSweep << <fullBlocksPerGrid, blockSize >> > (n, dev_odata, d);
				checkCUDAError("kernDownSweep failed!");
				cudaDeviceSynchronize();
			}

			cudaMemcpy(odata, dev_odata, n * sizeof(int), cudaMemcpyDeviceToHost);
			checkCUDAError("cudaMemcpy dev_odata to odata failed!");

			cudaFree(dev_odata);

			/*for (int i = 0; i < n; i++) {
				printf("%d ", odata[i]);
			}*/
            //timer().endGpuTimer();
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
        

		int compactPower2(int n, int* odata, const int* idata) {
			timer().startGpuTimer();
			// TODO
			int blockSize = 128;

			int* dev_tempArray;
			int* dev_idata;
			int* dev_odata;

			int* host_tempArray = new int[n];
			int* host_scanArray = new int[n];
			memset(host_tempArray, 0, n * sizeof(int));
			memset(host_scanArray, 0, n * sizeof(int));

			cudaMalloc((void**)&dev_tempArray, n * sizeof(int));
			checkCUDAError("cudaMalloc dev_tempArray failed!");
			cudaMalloc((void**)&dev_idata, n * sizeof(int));
			checkCUDAError("cudaMalloc dev_idata failed!");
			cudaMalloc((void**)&dev_odata, n * sizeof(int));
			checkCUDAError("cudaMalloc dev_odata failed!");

			cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);
			checkCUDAError("cudaMemcpy idata to dev_idata failed!");

			// compute tempArray
			computeTempArray << <(n + blockSize - 1) / blockSize, blockSize >> > (n, dev_tempArray, dev_idata);
			checkCUDAError("computeTempArray failed!");
			cudaDeviceSynchronize();

			cudaMemcpy(host_tempArray, dev_tempArray, n * sizeof(int), cudaMemcpyDeviceToHost);
			checkCUDAError("cudaMemcpy dev_tempArray to host_tempArray failed!");

			// up sweep and down sweep
			scan(n, host_scanArray, host_tempArray);

			// scatter
			int* dev_scanArray;
			cudaMalloc((void**)&dev_scanArray, n * sizeof(int));
			checkCUDAError("cudaMalloc dev_scanArray failed!");
			cudaMemcpy(dev_scanArray, host_scanArray, n * sizeof(int), cudaMemcpyHostToDevice);
			checkCUDAError("cudaMemcpy host_scanArray to dev_scanArray failed!");

			scatter << <(n + blockSize - 1) / blockSize, blockSize >> > (n, dev_odata, dev_idata, dev_tempArray, dev_scanArray);
			checkCUDAError("scatter failed!");
			cudaDeviceSynchronize();
			cudaMemcpy(odata, dev_odata, n * sizeof(int), cudaMemcpyDeviceToHost);

			timer().endGpuTimer();

			cudaFree(dev_tempArray);
			cudaFree(dev_idata);


			return host_scanArray[n - 1];
		}

		int compact(int n, int* odata, const int* idata) {
			int npower2 = 1 << ilog2ceil(n);
			int* idata_power2 = new int[npower2];
			memset(idata_power2, 0, npower2 * sizeof(int));
			memcpy(idata_power2, idata, n * sizeof(int));

			int* odata_power2 = new int[npower2];
			memset(odata_power2, 0, npower2 * sizeof(int));

			int count = compactPower2(npower2, odata_power2, idata_power2);
			memcpy(odata, odata_power2, count * sizeof(int));

			delete[] idata_power2;
			delete[] odata_power2;
			
			return count;
		}
    }
}

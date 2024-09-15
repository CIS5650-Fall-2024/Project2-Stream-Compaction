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

		// up-sweep kernel
        __global__ void kernUpSweep(int n, int* odata, const int* idata, int t) {
            int index = threadIdx.x + (blockIdx.x * blockDim.x);
            if (index >= n) {
                return;
            }
			// exclusive scan
			odata[index] = (index > 0) ? idata[index - 1] : 0;
			__syncthreads();
			// upsweep
            for (int d = 0; d <= t; ++d) {
                int offset = 1 << (d + 1);
				int ai = index + offset - 1;
				int bi = index + (offset / 2) - 1;
                if (index < n && (index % offset) == 0) {
                    odata[ai] += odata[bi];
                }

                __syncthreads();
            }
        }

        // down-sweep kernel
		__global__ void kernDownSweep(int n, int* odata, const int* idata, int t) {
			int index = threadIdx.x + (blockIdx.x * blockDim.x);
			if (index >= 1 << (t + 1)) {
				return;
			}
			// exclusive scan
			odata[index] = (index > 0) ? idata[index - 1] : 0;
			__syncthreads();
			// downsweep
			if (index == 0) {
				odata[n - 1] = 0;
			}
			for (int d = t; d >= 0; --d) {
				int offset = 1 << (d + 1);
				int ai = index + offset - 1;
				int bi = index + (offset / 2) - 1;
				if (index < n && (index % offset) == 0) {
					int temp = odata[bi];
					odata[bi] = odata[ai];
					odata[ai] += temp;
				}

				__syncthreads();
			}
		}

		// up sweep + down aweep
		__global__ void kernScan(int n, int* odata, const int* idata, int t) {
			int index = threadIdx.x + (blockIdx.x * blockDim.x);
			int paddedSize = 1 << (t + 1);
			if (index >= paddedSize) {
				return;
			}
			// exclusive scan
			//odata[index] = (index > 0) ? idata[index - 1] : 0;
			//odata[index] = idata[index];
			odata[index] = (index >= n) ? 0 : idata[index];
			__syncthreads();
			// upsweep
			for (int d = 0; d <= t; ++d) {
				int offset = 1 << (d + 1);
				int ai = index + offset - 1;
				int bi = index + (offset / 2) - 1;
				if (index < paddedSize && (index % offset) == 0) {
					odata[ai] += odata[bi];
				}

				__syncthreads();
			}
			// downsweep
			if (index == 0) {
				odata[paddedSize - 1] = 0;
			}
			
			for (int d = t; d >= 0; --d) {
				int offset = 1 << (d + 1);
				int ai = index + offset - 1;
				int bi = index + (offset / 2) - 1;
				if (index < paddedSize && (index % offset) == 0) {
					int temp = odata[bi];
					odata[bi] = odata[ai];
					odata[ai] += temp;
				}

				__syncthreads();
			}
		}


        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            timer().startGpuTimer();
            // TODO
			// call kernel
			int* dev_idata;
			int* dev_odata;
			cudaMalloc((void**)&dev_idata, n * sizeof(int));
			cudaMalloc((void**)&dev_odata, n * sizeof(int));
			cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            int t = ilog2ceil(n) - 1;
            printf("log2_n - 1: %d\n", t);
            //kernUpSweep << <1, n >> > (n, dev_odata, dev_idata, t);
			//kernDownSweep << <1, n >> > (n, dev_odata, dev_idata, t);
			kernScan << <1, n >> > (n, dev_odata, dev_idata, t);
			cudaMemcpy(odata, dev_odata, n * sizeof(int), cudaMemcpyDeviceToHost);
			cudaFree(dev_idata);
			cudaFree(dev_odata);




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
        int compact(int n, int *odata, const int *idata) {
            timer().startGpuTimer();
            // TODO
            timer().endGpuTimer();
            return -1;
        }
    }
}

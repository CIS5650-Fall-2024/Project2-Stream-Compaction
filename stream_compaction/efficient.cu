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

        __global__ void kernUpSweep(int n, int offset, int* x) {
            int index = threadIdx.x + (blockIdx.x * blockDim.x);
            if (index >= n) {
				return;
			}

            int k = index * offset;
            x[k + offset - 1] += x[k + offset / 2 - 1];
        }

        __global__ void kernDownSweep(int n, int offset, int* x) {
			int index = threadIdx.x + (blockIdx.x * blockDim.x);
            if (index >= n) {
                return;
            }

            int k = index * offset;
            int t = x[k + offset / 2 - 1];
            x[k + offset / 2 - 1] = x[k + offset - 1];
            x[k + offset - 1] += t;
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            // memory operation
            int max_d = ilog2ceil(n);
            int next_power_of_two = pow(2, max_d);

            int* x;
            cudaMalloc((void**)&x, next_power_of_two * sizeof(int));
            cudaMemcpy(x, idata, n * sizeof(int), cudaMemcpyHostToDevice);

            timer().startGpuTimer();

            // TODO
            int blockSize = 64;

            // up-sweep
            for (int d = 0; d < max_d; ++d) {
                int step = pow(2, d + 1);
                int threadCount = next_power_of_two / step;
                dim3 fullBlocksPerGrid((threadCount + blockSize - 1) / blockSize);

				kernUpSweep<<<fullBlocksPerGrid, blockSize >>>(threadCount, step, x);
			}

            // down-sweep
            cudaMemset(x + next_power_of_two - 1, 0, sizeof(int));
            for (int d = max_d - 1; d >= 0; --d) {
				int step = pow(2, d + 1);
                int threadCount = next_power_of_two / step;
				dim3 fullBlocksPerGrid((threadCount + blockSize - 1) / blockSize);

                kernDownSweep<<<fullBlocksPerGrid, blockSize >>>(threadCount, step, x);
            }

            timer().endGpuTimer();

            cudaMemcpy(odata, x, n * sizeof(int), cudaMemcpyDeviceToHost);

            cudaFree(x);
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
            int* bools, *scanArr, *out, *in;
            cudaMalloc((void**)&bools, n * sizeof(int));
            cudaMalloc((void**)&scanArr, n * sizeof(int));
            cudaMalloc((void**)&out, n * sizeof(int));
            cudaMalloc((void**)&in, n * sizeof(int));

            cudaMemcpy(in, idata, n * sizeof(int), cudaMemcpyHostToDevice);

            //timer().startGpuTimer();
            
            // TODO
            int blockSize = 64;
            dim3 fullBlocksPerGrid((n + blockSize - 1) / blockSize);

            // Step 1: Compute temporary array of 0s and 1s
            StreamCompaction::Common::kernMapToBoolean<<<fullBlocksPerGrid, blockSize >>>(n, bools, in);

            // Step2: Run exclusive scan on tempArr
            scan(n, scanArr, bools);

            // Step 3: Scatter
            StreamCompaction::Common::kernScatter<<<fullBlocksPerGrid, blockSize >>>(n, out, in, bools, scanArr);

            //timer().endGpuTimer();

            int count = 0, lastScan = 0;
            cudaMemcpy(&count, scanArr + n - 1, sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(&lastScan, bools + n - 1, sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(odata, out, n * sizeof(int), cudaMemcpyDeviceToHost);

            cudaFree(bools);
            cudaFree(scanArr);
            cudaFree(out);
            cudaFree(in);

            return count + lastScan;
        }
    }
}

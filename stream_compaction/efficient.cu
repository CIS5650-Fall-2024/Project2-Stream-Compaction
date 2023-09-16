#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

#define NUM_BANKS 16 
#define LOG_NUM_BANKS 4 
#define CONFLICT_FREE_OFFSET(n) ((n) >> NUM_BANKS + (n) >> (2 * LOG_NUM_BANKS)) 

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
            x[k + offset - 1] += x[k + (offset >> 1) - 1];
        }

        __global__ void kernDownSweep(int n, int offset, int* x) {
			int index = threadIdx.x + (blockIdx.x * blockDim.x);
            if (index >= n) {
                return;
            }

            int left = (1 + (index << 1)) * offset - 1,
                right = left + offset;

            int t = x[left];
            x[left] = x[right];
            x[right] += t;
        }

        __global__ void kernPreScan(int n, int* odata, const int* idata) {
            extern __shared__ int temp[];

			int index = threadIdx.x + (blockIdx.x * blockDim.x);
            if (index >= n) {
				return;
			}

            int offset = 1, ai = index, bi = index + n << 1,
                bankOffsetA = CONFLICT_FREE_OFFSET(ai), bankOffsetB = CONFLICT_FREE_OFFSET(bi);

            temp[ai + bankOffsetA] = idata[ai];
            temp[bi + bankOffsetB] = idata[bi];

            // up sweep
            for (int d = n >> 1; d > 0; d >>= 1) {
	        	__syncthreads();
   
                if (index < d) {
                    int ai = ((index << 1) + 1) * offset - 1;
                    int bi = ai + offset;

                    ai += CONFLICT_FREE_OFFSET(ai);
                    bi += CONFLICT_FREE_OFFSET(bi);

                    temp[bi] += temp[ai];
	       	    }
           
	       	    offset <<= 1;
	        }
   
            // clear the last element before down sweep
            if (index == 0) {
	 	        temp[n - 1 + CONFLICT_FREE_OFFSET(n - 1)] = 0;
	        }
   
            // down sweep
            for (int d = 1; d < n; d <<= 1) {
                offset >>= 1;
      
                __syncthreads();
                if (index < d) {
                    int ai = ((index << 1) + 1) * offset - 1;
                    int bi = ai + offset;

                    ai += CONFLICT_FREE_OFFSET(ai);
                    bi += CONFLICT_FREE_OFFSET(bi);
      
                    int t = temp[ai];
                    temp[ai] = temp[bi];
                    temp[bi] += t;
                }
            }
   
            __syncthreads();
   
            odata[ai] = temp[ai + bankOffsetA];
            odata[bi] = temp[bi + bankOffsetB];
		}

        void scanShared(int n, int* odata, const int* idata) {
            int max_d = ilog2ceil(n);
            int next_power_of_two = 1 << max_d;

            int* in, int* out;
            cudaMalloc((void**)&in, next_power_of_two * sizeof(int));
            cudaMalloc((void**)&out, n * sizeof(int));
            cudaMemcpy(in, idata, n * sizeof(int), cudaMemcpyHostToDevice);

            timer().startGpuTimer();
            kernPreScan<<<1, next_power_of_two << 1, next_power_of_two * sizeof(int) >> >(next_power_of_two, out, in);
            timer().endGpuTimer();

            cudaMemcpy(odata, out, n * sizeof(int), cudaMemcpyDeviceToHost);
            cudaFree(in);
            cudaFree(out);
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            // memory operation
            int max_d = ilog2ceil(n);
            int next_power_of_two = 1 << max_d;

            int* x;
            cudaMalloc((void**)&x, next_power_of_two * sizeof(int));
            cudaMemcpy(x, idata, n * sizeof(int), cudaMemcpyHostToDevice);

            timer().startGpuTimer();

            // TODO
            int blockSize = 64, step = 1, threadCount = next_power_of_two;

            // up-sweep
            for (int d = 0; d < max_d; ++d) {
                step <<= 1;
                threadCount >>= 1;

                int curBlockSize = std::min(threadCount, blockSize);
                dim3 fullBlocksPerGrid((threadCount + curBlockSize - 1) / curBlockSize);

				kernUpSweep<<<fullBlocksPerGrid, curBlockSize >>>(threadCount, step, x);
			}

            // down-sweep
            cudaMemset(x + next_power_of_two - 1, 0, sizeof(int));

            for (int d = max_d - 1; d >= 0; --d) {
                step >>= 1;
                
                int curBlockSize = std::min(threadCount, blockSize);
                dim3 fullBlocksPerGrid((threadCount + curBlockSize - 1) / curBlockSize);

                kernDownSweep<<<fullBlocksPerGrid, curBlockSize >>>(threadCount, step, x);
                threadCount <<= 1;
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

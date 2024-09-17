#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"

# define RECURSIVE_SCAN 0		

namespace StreamCompaction {
    namespace Naive {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }
        // TODO: __global__
		int getLog2(int n) {
			int log2 = 0;
			while (n >>= 1) {
				log2++;
			}
			return log2+1;
		}

		__global__ void kernInclusiveScan(int n, int* odata, const int* idata, int t) {
			int index = threadIdx.x + (blockIdx.x * blockDim.x);
			
			if (index >= n) {
				return;
			}

			odata[index] = idata[index];
			__syncthreads();

			if(index >= t) {
				odata[index] = idata[index - t] + odata[index];
			}
		}

		__global__ void kernScan(int n, int* odata, const int* idata, int log2_n) {
			int index = (blockIdx.x * blockDim.x) + threadIdx.x;
			int pedding = 1 << log2_n;
			if (index >= pedding) {
				return;
			}
			// exclusive scan
			odata[index] = (index > 0) ? idata[index - 1] : 0;
			__syncthreads(); // odata with first element as 0 is ready

			for (int d = 1; d <= log2_n; ++d) {
				int t = 1 << (d - 1);
				int temp = 0;
				if (index >= t) temp = odata[index - t];  // Load the previous step's value
				__syncthreads();  // Synchronize before updating
				if (index >= t) odata[index] += temp;  // Update the current value
				__syncthreads();  // Synchronize after updating
			}
		}

		__global__ void kernBlockWiseExclusiveScan(int n, int* odata, const int* idata, int blockSize) {
			extern __shared__ int sdata[];

			int idx = threadIdx.x;
			int blockStartIndex = blockIdx.x * blockDim.x;
			int index = blockStartIndex + idx;

			// Load data into shared memory
			if (index < n) {
				sdata[idx] = idata[index];//(idx > 0) ? idata[index - 1] : 0;
			}
			else {
				sdata[idx] = 0;  // Out-of-range threads
			}
			__syncthreads();

			// Perform in-block scan
			for (int d = 1; d < blockDim.x; d *= 2) {
				int t = idx >= d ? sdata[idx - d] : 0;
				__syncthreads();
				if (idx >= d) {
					sdata[idx] += t;
				}
				__syncthreads();
			}

			// Write results to global memory
			if (index < n) {
				odata[index] = sdata[idx];
			}
		}



		// kernel for write total sum of each block into a new array
		__global__ void kernWriteBlockSum(int n, const int* odata, int* blockSum) {
			int index = threadIdx.x + (blockIdx.x * blockDim.x);
			if (index >= n) {
				return;
			}

			if ((index + 1) % (blockDim.x) == 0) {
				int i = (index + 1) / (blockDim.x) - 1;
				blockSum[i] = odata[index];
			}
		}

		// kernel for add block increments to each element in the corresponding block
		__global__ void kernAddBlockSum(int n, int* odata, const int* blockSum) {
			int index = threadIdx.x + (blockIdx.x * blockDim.x);
			if (index >= n) {
				return;
			}

			// exclusive scan
			// set first element to 0
			
			int temp = 0;
			if (index == 0) {
				temp = 0;
			}
			else {
				int blockIdx = (index - 1) / blockDim.x;
				int sumToAdd = blockSum[blockIdx];
				temp = odata[index - 1] + sumToAdd;
			}
			odata[index] = temp;
		}

		

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
			// TODO
			int log2_n = getLog2(n);
			int t = ilog2ceil(n);
			//printf("n: %d\n", n);
			printf("log2_n: %d\n", t);
			int blockSize = 256;
			int numBlocks = (n + blockSize - 1) / blockSize;
			dim3 fullBlocksPerGrid(numBlocks);
			printf("block size: %d\n", blockSize);
			printf("numBlocks: %d\n", numBlocks);
			// get block size and block number for scan block sum
			int numBlocks_scan = (numBlocks + blockSize - 1) / blockSize;
			dim3 fullBlocksPerGrid_scan(numBlocks_scan);
			
			int* dev_idata;
			int* dev_odata;
			int* dev_blockSum;
			int* dev_blockIncrements;
			int* blockSum = new int[n];
			cudaMalloc((void**)&dev_idata, n * sizeof(int));
			cudaMalloc((void**)&dev_odata, n * sizeof(int));
			cudaMalloc((void**)&dev_blockSum, n * sizeof(int));
			cudaMalloc((void**)&dev_blockIncrements, n * sizeof(int));
			cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);
			cudaMemcpy(dev_blockSum, blockSum, blockSize * sizeof(int), cudaMemcpyHostToDevice);

			timer().startGpuTimer();
			// call kernel
# if RECURSIVE_SCAN
			// scan on each block
			//kernScan << <fullBlocksPerGrid, blockSize >> > (n, dev_odata, dev_idata, t); // non-shared memory one
			kernBlockWiseExclusiveScan << <fullBlocksPerGrid, blockSize, blockSize * sizeof(int) >> > (n, dev_odata, dev_idata, blockSize); // shared memory one
			// write total sum of each block to blockSum
			kernWriteBlockSum << <fullBlocksPerGrid, blockSize >> > (n, dev_odata, dev_blockSum);
			// scan on blockSum
			int blockSumSize = ilog2ceil(numBlocks);
			kernScan << <1, numBlocks >> > (numBlocks, dev_blockIncrements, dev_blockSum, blockSumSize);
			//kernScan << <fullBlocksPerGrid_scan, blockSize >> > (numBlocks, dev_blockIncrements, dev_blockSum, blockSumSize);
			// recursive scan on blockSum
			// add block increments to each element in the corresponding block
			kernAddBlockSum << <fullBlocksPerGrid, blockSize >> > (n, dev_odata, dev_blockIncrements);
			//dev_odata = dev_blockIncrements; // for testing
			timer().endGpuTimer();
			cudaMemcpy(odata, dev_odata, n * sizeof(int), cudaMemcpyDeviceToHost);
# else
			for (int d = 1; d <= log2_n; d++) {
				int pedding = 1 << (d - 1);
				kernInclusiveScan << <fullBlocksPerGrid, blockSize >> > (n, dev_odata, dev_idata, pedding);
				int* temp = dev_idata;
				dev_idata = dev_odata;
				dev_odata = temp;			
			}


			timer().endGpuTimer();

			// right shift odata
			odata[0] = 0;
			cudaMemcpy(odata + 1, dev_idata, (n - 1) * sizeof(int), cudaMemcpyDeviceToHost);

#endif
			
			cudaFree(dev_idata);
			cudaFree(dev_odata);
			cudaFree(dev_blockSum);
			delete[] blockSum;

          
        }
    }
}

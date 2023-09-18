#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

__global__ void kernUpSweep(int n, int d, int* data) {
    int index = threadIdx.x + (blockIdx.x * blockDim.x);
    if (index >= n) {
        return;
    }
    int k = (index + 1) * (1 << (d + 1)) - 1;
    data[k] += data[k - (1 << d)];
}

__global__ void kernDownSweep(int n, int d, int* data) {
    int index = threadIdx.x + (blockIdx.x * blockDim.x);
    if (index >= n) {
        return;
    }
    int k = (index + 1) * (1 << (d + 1)) - 1;
    int val = data[k - (1 << d)];
    data[k - (1 << d)] = data[k];    
    data[k] += val;
}

namespace StreamCompaction {
    namespace Efficient {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            int* dev_data;
            int noOfIters = ilog2ceil(n) - 1;
            
            cudaMalloc((void**)&dev_data, n * sizeof(int));
            checkCUDAError("cudaMalloc dev_data failed!");

            cudaMemcpy(dev_data, idata, sizeof(int) * n, cudaMemcpyHostToDevice);

            timer().startGpuTimer();
            
            //up sweep
            for (int d = 0; d <= noOfIters; d++) {
                int noOfElementsToBeUpdated = n / (1 << (d + 1));
                dim3 fullBlocksPerGrid((noOfElementsToBeUpdated + BLOCKSIZE - 1) / BLOCKSIZE);
                kernUpSweep << <fullBlocksPerGrid, BLOCKSIZE >> > (n, d, dev_data);
            }

            //set last element to zero before starting down sweep
            int zero = 0;
            cudaMemcpy(&dev_data[n-1], &zero, sizeof(int), cudaMemcpyHostToDevice);

            //down sweep
            for (int d = noOfIters; d >= 0; d--) {
                int noOfElementsToBeUpdated = n / (1 << (d + 1));
                dim3 fullBlocksPerGrid((noOfElementsToBeUpdated + BLOCKSIZE - 1) / BLOCKSIZE);
                kernDownSweep << <fullBlocksPerGrid, BLOCKSIZE >> > (n, d, dev_data);
            }
            cudaMemcpy(odata, dev_data, sizeof(int) * n, cudaMemcpyDeviceToHost);
            timer().endGpuTimer();
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
            timer().startGpuTimer();
            // TODO
            timer().endGpuTimer();
            return -1;
        }
    }
}

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
        void scan(int n, int *odata, const int *idata, int BLOCKSIZE) {
            int* dev_data;
            int paddedSize = 1 << ilog2ceil(n);
            int noOfIters = ilog2ceil(paddedSize) - 1;
            
            cudaMalloc((void**)&dev_data, paddedSize * sizeof(int));
            checkCUDAError("cudaMalloc dev_data failed!");

            cudaMemcpy(dev_data, idata, sizeof(int) * n, cudaMemcpyHostToDevice);
            cudaMemset(&dev_data[n], 0, sizeof(int) * (paddedSize - n));
            timer().startGpuTimer();
            
            //up sweep
            for (int d = 0; d <= noOfIters; d++) {
                int noOfElementsToBeUpdated = paddedSize / (1 << (d + 1));
                dim3 fullBlocksPerGrid((noOfElementsToBeUpdated + BLOCKSIZE - 1) / BLOCKSIZE);
                kernUpSweep << <fullBlocksPerGrid, BLOCKSIZE >> > (noOfElementsToBeUpdated, d, dev_data);
            }

            //set last element to zero before starting down sweep            
            cudaMemset(&dev_data[paddedSize -1], 0, sizeof(int));

            //down sweep
            for (int d = noOfIters; d >= 0; d--) {
                int noOfElementsToBeUpdated = paddedSize / (1 << (d + 1));
                dim3 fullBlocksPerGrid((noOfElementsToBeUpdated + BLOCKSIZE - 1) / BLOCKSIZE);
                kernDownSweep << <fullBlocksPerGrid, BLOCKSIZE >> > (noOfElementsToBeUpdated, d, dev_data);
            }
            timer().endGpuTimer();
            cudaMemcpy(odata, dev_data, sizeof(int) * n, cudaMemcpyDeviceToHost);
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
        int compact(int n, int *odata, const int *idata, int BLOCKSIZE) {
            int* dev_idata;
            int* dev_odata;
            int* dev_bools;
            int* dev_indices;

            int paddedSize = 1 << ilog2ceil(n);
            int noOfIters = ilog2ceil(paddedSize) - 1;

            cudaMalloc((void**)&dev_idata, paddedSize * sizeof(int));
            checkCUDAError("cudaMalloc dev_idata failed!");
            cudaMemcpy(dev_idata, idata, sizeof(int) * n, cudaMemcpyHostToDevice);
            cudaMemset(&dev_idata[n], 0, sizeof(int) * (paddedSize - n));

            cudaMalloc((void**)&dev_odata, paddedSize * sizeof(int));
            checkCUDAError("cudaMalloc dev_odata failed!");

            cudaMalloc((void**)&dev_bools, paddedSize * sizeof(int));
            checkCUDAError("cudaMalloc dev_bools failed!");

            cudaMalloc((void**)&dev_indices, paddedSize * sizeof(int));
            checkCUDAError("cudaMalloc dev_indices failed!");

            timer().startGpuTimer();

            //Map to booleans
            dim3 fullBlocksPerGrid((paddedSize + BLOCKSIZE - 1) / BLOCKSIZE);
            StreamCompaction::Common::kernMapToBoolean << <fullBlocksPerGrid, BLOCKSIZE >> > (paddedSize, dev_bools, dev_idata);
            
            cudaMemcpy(dev_indices, dev_bools, sizeof(int) * paddedSize, cudaMemcpyDeviceToDevice);

            //scan

            //up sweep
            for (int d = 0; d <= noOfIters; d++) {
                int noOfElementsToBeUpdated = paddedSize / (1 << (d + 1));
                dim3 fullBlocksPerGrid((noOfElementsToBeUpdated + BLOCKSIZE - 1) / BLOCKSIZE);
                kernUpSweep << <fullBlocksPerGrid, BLOCKSIZE >> > (noOfElementsToBeUpdated, d, dev_indices);
            }
            
            //set last element to zero before starting down sweep
            cudaMemset(&dev_indices[paddedSize - 1], 0, sizeof(int));
            
            //down sweep
            for (int d = noOfIters; d >= 0; d--) {
                int noOfElementsToBeUpdated = paddedSize / (1 << (d + 1));
                dim3 fullBlocksPerGrid((noOfElementsToBeUpdated + BLOCKSIZE - 1) / BLOCKSIZE);
                kernDownSweep << <fullBlocksPerGrid, BLOCKSIZE >> > (noOfElementsToBeUpdated, d, dev_indices);
            }
            
            int returnVal;
            cudaMemcpy(&returnVal, dev_indices + paddedSize - 1, sizeof(int), cudaMemcpyDeviceToHost);

            //scatter
            StreamCompaction::Common::kernScatter << <fullBlocksPerGrid, BLOCKSIZE >> > (paddedSize, dev_odata, dev_idata, dev_bools, dev_indices);
            timer().endGpuTimer();
            cudaMemcpy(odata, dev_odata, sizeof(int) * returnVal, cudaMemcpyDeviceToHost);                    

            //cleanup
            cudaFree(dev_idata);
            cudaFree(dev_odata);
            cudaFree(dev_bools);
            cudaFree(dev_indices); 
            return returnVal;
        }
    }
}

#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"
#include <iostream>

namespace StreamCompaction {
    namespace Efficient {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        __global__ void upSweep(int n, int* data, int step) {
            int index = threadIdx.x + (blockIdx.x * blockDim.x);
            if (index < n / step) {
                index *= step;
                data[index + step - 1] += data[index + (step >> 1) - 1];
            }
        }

        __global__ void downSweep(int n, int* data, int step) {
            int index = threadIdx.x + (blockIdx.x * blockDim.x);
            if (index < n / step) {
                index *= step;
                int leftChildIndex = index + (step >> 1) - 1;
                int rightChildIndex = index + step - 1;
                int temp = data[leftChildIndex];
                data[leftChildIndex] = data[rightChildIndex];
                data[rightChildIndex] += temp;
            }
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            int* dev_idata;
        
            int numLevels = ilog2ceil(n);
            int powerOf2Size = 1 << numLevels;
            int numZerosToSet = powerOf2Size - n;

            cudaMalloc((void**)&dev_idata, powerOf2Size * sizeof(int));
            checkCUDAError("cudaMalloc dev_idata failed!");
            cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            cudaMemset(&dev_idata[n], 0, numZerosToSet * sizeof(int));

            timer().startGpuTimer();

            // Step 1: Up Sweep
            for (int d = 0; d < numLevels; ++d) {
                int numThreads = powerOf2Size / (1 << (d + 1));
                dim3 BlocksPerGrid((numThreads + blockSize - 1) / blockSize);
                upSweep <<<BlocksPerGrid, blockSize>>> (powerOf2Size, dev_idata, 1 << (d + 1));
            }
            cudaDeviceSynchronize();

            // Step 2: Down Sweep
            cudaMemset(&dev_idata[powerOf2Size - 1], 0, sizeof(int));
            for (int d =  numLevels - 1; d >= 0; --d) {
                int numThreads = powerOf2Size / (1 << (d + 1));
                dim3 BlocksPerGrid((numThreads + blockSize - 1) / blockSize);
                downSweep <<<BlocksPerGrid, blockSize>>> (powerOf2Size, dev_idata, 1 << (d + 1));
            }
            
            timer().endGpuTimer();

            cudaDeviceSynchronize();
            cudaMemcpy(odata, dev_idata, n * sizeof(int), cudaMemcpyDeviceToHost);
            cudaFree(dev_idata);
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
            int numLevels = ilog2ceil(n);
            int powerOf2Size = 1 << numLevels;
            int numZerosToSet = powerOf2Size - n;

            cudaMalloc((void**)&dev_idata, powerOf2Size * sizeof(int));
            checkCUDAError("cudaMalloc dev_idata failed!");
            cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            cudaMemset(&dev_idata[n], 0, numZerosToSet * sizeof(int));

            int* dev_bools;
            cudaMalloc((void**)&dev_bools, powerOf2Size * sizeof(int));
            checkCUDAError("cudaMalloc dev_bools failed!");

            int* dev_indices;
            cudaMalloc((void**)&dev_indices, powerOf2Size * sizeof(int));
            checkCUDAError("cudaMalloc dev_indices failed!");

            int* dev_odata;
            cudaMalloc((void**)&dev_odata, powerOf2Size * sizeof(int));
            checkCUDAError("cudaMalloc dev_odata failed!");

            dim3 fullBlocksPerGrid((powerOf2Size + blockSize - 1) / blockSize);

            timer().startGpuTimer();

            //Step 1: Compute temporary array
            StreamCompaction::Common::kernMapToBoolean <<<fullBlocksPerGrid, blockSize>>> (powerOf2Size, dev_bools, dev_idata);
            cudaDeviceSynchronize();
            cudaMemcpy(dev_indices, dev_bools, powerOf2Size * sizeof(int), cudaMemcpyDeviceToDevice);

            //Step 2: Run exclusive scan on temp array
            //Had to copy code from scan to avoid duplicate timer
            //Step 2.1: Up Sweep
            for (int d = 0; d < numLevels; ++d) {
                int numThreads = powerOf2Size / (1 << (d + 1));
                dim3 BlocksPerGrid((numThreads + blockSize - 1) / blockSize);
                upSweep <<<BlocksPerGrid, blockSize>>> (powerOf2Size, dev_indices, 1 << (d + 1));
            }
            cudaDeviceSynchronize();
            //Step 2.2: Down Sweep
            cudaMemset(&dev_indices[powerOf2Size - 1], 0, sizeof(int));
            for (int d = numLevels - 1; d >= 0; --d) {
                int numThreads = powerOf2Size / (1 << (d + 1));
                dim3 BlocksPerGrid((numThreads + blockSize - 1) / blockSize);
                downSweep <<<BlocksPerGrid, blockSize>>> (powerOf2Size, dev_indices, 1 << (d + 1));
            }
            cudaDeviceSynchronize();

            //Step 3: Scatter
            StreamCompaction::Common::kernScatter <<<fullBlocksPerGrid, blockSize>>> (powerOf2Size, dev_odata, dev_idata, dev_bools, dev_indices);
            cudaDeviceSynchronize();
            timer().endGpuTimer();

            int dev_indicesLastElem;
            int dev_boolsLastElem;
            cudaMemcpy(&dev_indicesLastElem, &dev_indices[powerOf2Size - 1], sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(&dev_boolsLastElem, &dev_bools[powerOf2Size - 1], sizeof(int), cudaMemcpyDeviceToHost);

            int numElements = dev_indicesLastElem + dev_boolsLastElem;
            cudaMemcpy(odata, dev_odata, numElements * sizeof(int), cudaMemcpyDeviceToHost);

            cudaFree(dev_idata);
            cudaFree(dev_bools);
            cudaFree(dev_indices);
            cudaFree(dev_odata);

            return numElements;
        }
    }
}

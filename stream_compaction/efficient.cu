#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"


__global__ void upsweep(int n, int d, int* data) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = 1 << (d + 1);

    //validation check
    if (index >= n) {
        return;
    }

    if (index % stride == 0) {
        data[index + stride - 1] += data[index + (stride >> 1) - 1];
    }
}


__global__ void downsweep(int n, int d, int* data) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = 1 << (d + 1);

    //validation check
    if (index >= n) {
        return;
    }


    if (index % stride == 0) {
        int left = index + (stride >> 1) - 1;
        int right = index + stride - 1;
        int temp = data[left];
        data[left] = data[right];
        data[right] += temp;
    }
}


__global__ void computePredicate(int n, const int* idata, int* predicate) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index >= n) return;

    predicate[index] = (idata[index] != 0) ? 1 : 0;
}

__global__ void scatter(int n, const int* idata, int* odata, const int* scanned) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index >= n) return;

    if (idata[index] != 0) {
        odata[scanned[index]] = idata[index];
    }
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

        void scan(int n, int* odata, const int* idata) {

            // Allocate memory on the GPU
            int* dev_data;
            cudaMalloc((void**)&dev_data, n * sizeof(int));
            cudaMemcpy(dev_data, idata, n * sizeof(int), cudaMemcpyHostToDevice);

            // Set up the grid and block size
            int blockSize = 128;
            int numBlocks = (n + blockSize - 1) / blockSize;

            // Start GPU timer
            timer().startGpuTimer(); // Start the timer once here

            // Up-sweep phase
            for (int d = 0; (1 << d) < n; ++d) {
                upsweep <<<numBlocks, blockSize>>> (n, d, dev_data);
                cudaDeviceSynchronize(); // Wait for GPU to finish current kernel
            }

            // Set the last element to zero
            cudaMemset(dev_data + n - 1, 0, sizeof(int));

            // Down-sweep phase
            for (int d = int(log2(n)) - 1; d >= 0; --d) {
                downsweep <<<numBlocks, blockSize>>> (n, d, dev_data);
                cudaDeviceSynchronize(); // Wait for GPU to finish current kernel
            }

            // End GPU timer
            timer().endGpuTimer(); // Stop the timer once here

            // Copy results back to host
            cudaMemcpy(odata, dev_data, n * sizeof(int), cudaMemcpyDeviceToHost);

            // Free GPU memory
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

        int compact(int n, int* odata, const int* idata) {
            // TODO
            // Allocate memory on the device
            int* dev_idata, * dev_odata, * dev_predicate, * dev_scanned;
            cudaMalloc((void**)&dev_idata, n * sizeof(int));
            cudaMalloc((void**)&dev_odata, n * sizeof(int));
            cudaMalloc((void**)&dev_predicate, n * sizeof(int));
            cudaMalloc((void**)&dev_scanned, n * sizeof(int));

            cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);

            int blockSize = 128;
            int numBlocks = (n + blockSize - 1) / blockSize;

            timer().startGpuTimer();

            // Compute predicate
            computePredicate <<<numBlocks, blockSize>>> (n, dev_idata, dev_predicate);
            cudaDeviceSynchronize();

            // Perform exclusive scan on the predicate array
            scan(n, dev_scanned, dev_predicate);

            // Scatter the valid elements to the output array
            scatter <<<numBlocks, blockSize>>> (n, dev_idata, dev_odata, dev_scanned);
            
            timer().endGpuTimer();

            cudaDeviceSynchronize();

            // Copy the results back to the host
            cudaMemcpy(odata, dev_odata, n * sizeof(int), cudaMemcpyDeviceToHost);

            // Compute the number of non-zero elements
            int count;
            cudaMemcpy(&count, dev_scanned + n - 1, sizeof(int), cudaMemcpyDeviceToHost);
            count += (idata[n - 1] != 0) ? 1 : 0; // Include the last element if it is non-zero

            cudaFree(dev_idata);
            cudaFree(dev_odata);
            cudaFree(dev_predicate);
            cudaFree(dev_scanned);

            return count; // Return the number of elements remaining after compaction
        }



    }
}

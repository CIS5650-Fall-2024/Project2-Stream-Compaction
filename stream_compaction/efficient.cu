#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"
#include <vector>

namespace StreamCompaction {
    namespace Efficient {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        // Kernel for the upsweep phase of the scan
        __global__ void kern_upsweep(int* data, int offset, int n) {
            int tid = threadIdx.x + blockIdx.x * blockDim.x;
            int index = tid * offset * 2;

            if (index + offset * 2 - 1 < n) {
                data[index + offset * 2 - 1] += data[index + offset - 1];
            }
        }

        // Kernel for the downsweep phase of the scan
        __global__ void kern_downsweep(int* data, int offset, int n) {
            int tid = threadIdx.x + blockIdx.x * blockDim.x;
            int index = tid * offset * 2;

            if (index + offset * 2 - 1 < n) {
                int temp = data[index + offset - 1];
                data[index + offset - 1] = data[index + offset * 2 - 1];
                data[index + offset * 2 - 1] += temp;
            }
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            int rounded_size = pow(2, ilog2ceil(n));

            // Allocate memory for device input/output arrays
            int* dev_data;
            cudaMalloc((void**)&dev_data, rounded_size * sizeof(int));
            checkCUDAError("cudaMalloc dev_data failed!");

            // Copy input data to device, with padding for non-power-of-two sizes
            cudaMemcpy(dev_data, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            checkCUDAError("cudaMemcpy dev_data failed");

            if (rounded_size > n) {
                cudaMemset(dev_data + n, 0, (rounded_size - n) * sizeof(int));
                checkCUDAError("cudaMemset dev_data failed");
            }

            timer().startGpuTimer();

            // Perform the upsweep phase
            for (int offset = 1; offset < rounded_size; offset *= 2) {
                int numBlocks = (rounded_size / (offset * 2) + blockSize - 1) / blockSize;
                kern_upsweep << <numBlocks, blockSize >> > (dev_data, offset, rounded_size);
                checkCUDAError("kern_upsweep failed!");
                cudaDeviceSynchronize();
            }

            // Set the last element to 0 (this is required by the downsweep phase)
            cudaMemset(dev_data + rounded_size - 1, 0, sizeof(int));
            checkCUDAError("cudaMemset dev_data failed");

            // Perform the downsweep phase
            for (int offset = rounded_size / 2; offset >= 1; offset /= 2) {
                int numBlocks = (rounded_size / (offset * 2) + blockSize - 1) / blockSize;
                kern_downsweep << <numBlocks, blockSize >> > (dev_data, offset, rounded_size);
                checkCUDAError("kern_downsweep failed!");
                cudaDeviceSynchronize();
            }
            
            timer().endGpuTimer();

            // Copy the result back to host
            cudaMemcpy(odata, dev_data, n * sizeof(int), cudaMemcpyDeviceToHost);
            checkCUDAError("cudaMemcpy odata failed");

            // Free device memory
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
            size_t paddedSize = (size_t) 1 << ilog2ceil(n);

            int* dev_idata;
            int* dev_odata;
            int* dev_bools;
            int* dev_indices;

            // Allocate memory for device arrays
            checkCUDAError("failed");
            cudaMalloc((void**)&dev_idata, paddedSize * sizeof(int));
            checkCUDAError("cudaMalloc dev_idata failed");
            cudaMalloc((void**)&dev_odata, paddedSize * sizeof(int));
            checkCUDAError("cudaMalloc dev_odata failed");
            cudaMalloc((void**)&dev_bools, paddedSize * sizeof(int));
            checkCUDAError("cudaMalloc dev_bools failed");
            cudaMalloc((void**)&dev_indices, paddedSize * sizeof(int));
            checkCUDAError("cudaMalloc dev_indices failed");

            // Copy input data to device
            cudaMemcpy(dev_idata, idata, sizeof(int) * n, cudaMemcpyHostToDevice);
            checkCUDAError("cudaMemcpy dev_idata failed");

            if (paddedSize > n) {
                cudaMemset(dev_idata + n, 0, (paddedSize - n) * sizeof(int));
                checkCUDAError("cudaMemset dev_data failed");
            }

            dim3 fullBlocksPerGrid((paddedSize + blockSize - 1) / blockSize);

            timer().startGpuTimer();
            
            // Step 1: Map to Boolean
            StreamCompaction::Common::kernMapToBoolean << <fullBlocksPerGrid, blockSize >> > (paddedSize, dev_bools, dev_idata);
            checkCUDAError("kernMapToBoolean failed");

            // Step 2: Perform Scan on Boolean Array
            cudaMemcpy(dev_indices, dev_bools, sizeof(int) * paddedSize, cudaMemcpyDeviceToDevice);

            // Up-sweep phase
            for (int offset = 1; offset < paddedSize; offset *= 2) {
                int numBlocks = (paddedSize / (offset * 2) + blockSize - 1) / blockSize;
                if (numBlocks > 0) { // Only run if there is work to do
                    kern_upsweep << <numBlocks, blockSize >> > (dev_indices, offset, paddedSize);
                    checkCUDAError("kern_upsweep failed!");
                    cudaDeviceSynchronize();
                }
            }

            // Set the last element to 0 (this is required by the downsweep phase)
            cudaMemset(dev_indices + paddedSize - 1, 0, sizeof(int));
            checkCUDAError("cudaMemset failed");

            // Down-sweep phase
            for (int offset = 1; offset < paddedSize; offset *= 2) {
                int numBlocks = (paddedSize / (offset * 2) + blockSize - 1) / blockSize;
                if (numBlocks > 0) { // Only run if there is work to do
                    kern_downsweep << <numBlocks, blockSize >> > (dev_indices, offset, paddedSize);
                    checkCUDAError("kern_downsweep failed");
                    cudaDeviceSynchronize();
                }
            }

            // Step 3: Scatter
            StreamCompaction::Common::kernScatter << <fullBlocksPerGrid, blockSize >> > (paddedSize, dev_odata, dev_idata, dev_bools, dev_indices);
            checkCUDAError("kernScatter failed");

            timer().endGpuTimer();

            // Step 4: Copy results and free memory
            std::vector<int> a;
            a.resize(paddedSize);
            cudaMemcpy(a.data(), dev_indices, paddedSize * sizeof(int), cudaMemcpyDeviceToHost);
            checkCUDAError("test failed");



            int compactedSize;
            cudaMemcpy(&compactedSize, dev_indices + paddedSize - 1, sizeof(int), cudaMemcpyDeviceToHost);
            checkCUDAError("cudaMemcpy compactedSize failed");
            cudaMemcpy(odata, dev_odata, sizeof(int) * compactedSize, cudaMemcpyDeviceToHost);
            checkCUDAError("cudaMemcpy odata failed");

            // Free memory
            cudaFree(dev_idata);
            cudaFree(dev_odata);
            cudaFree(dev_bools);
            cudaFree(dev_indices);


            return compactedSize;
        }
    }
}

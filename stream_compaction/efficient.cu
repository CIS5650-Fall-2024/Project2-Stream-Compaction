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

        // Performs ONE iteration of up sweep
        __global__ void kernUpSweep(int n, int d, int* data) {
            int index = blockIdx.x * blockDim.x + threadIdx.x;
            if (index >= n) {
                return;
            }

            int offset1 = 1 << d;
            int offset2 = 1 << (1 + d);
            if (index % offset2 == 0) {
                data[index + offset2 - 1] +=
                    data[index + offset1 - 1];
            }
        }

        __global__ void kernUpSweepOptimized(int n, int d, int stride, int* data) {
            int index = blockIdx.x * blockDim.x + threadIdx.x;
            if (index >= n) {
                return;
            }

            int offset = 1 << d;
            if (index < stride) {
                data[(index * 2 + 2) * offset - 1] +=
                    data[(index * 2 + 1) * offset - 1];
            }
        }

        // Performs ONE iteration of down sweep
        __global__ void kernDownSweep(int n, int d, int *data) {
            int index = blockIdx.x * blockDim.x + threadIdx.x;
            if (index >= n) {
                return;
            }

            int offset1 = 1 << d;
            int offset2 = 1 << (1 + d);
            if (index % offset2 == 0) {
                int left = data[index + offset1 - 1];                   // Save left child
                data[index + offset1 - 1] = data[index + offset2 - 1];  // Set left child to this node’s value
                data[index + offset2 - 1] += left;                      // Set right child to old left value +
                                                                        // this node’s value
            }
        }

        __global__ void kernDownSweepOptimized(int n, int d, int stride, int* data) {
            int index = blockIdx.x * blockDim.x + threadIdx.x;
            if (index >= n) {
                return;
            }

            int offset1 = (index * 2 + 1) * stride - 1;
            int offset2 = (index * 2 + 2) * stride - 1;
            if (index < d) {
                int left = data[offset1];       // Save left child
                data[offset1] = data[offset2];  // Set left child to this node’s value
                data[offset2] += left;          // Set right child to old left value +
                                                // this node’s value
            }
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            int* dev_buf;

            int power2 = ilog2ceil(n);
            int chunk = 1 << power2;

            dim3 blocksPerGrid((chunk + blockSize - 1) / blockSize);
            size_t arrSize = n * sizeof(int);

            cudaMalloc((void**)&dev_buf, chunk * sizeof(int));
            //checkCUDAError("cudaMalloc dev_buf failed!");

            cudaMemcpy(dev_buf, idata, arrSize, cudaMemcpyHostToDevice);
            //checkCUDAError("cudaMemcpy idata to dev_buf failed!");
            if (chunk > n) {
                cudaMemset(dev_buf + n, 0, (chunk - n) * sizeof(int));
                //checkCUDAError("cudaMemset dev_buf[n] failed!");
            }

            timer().startGpuTimer();
            // Up Sweep
            for (int d = 0; d <= power2 - 1; ++d) {
                kernUpSweep << <blocksPerGrid, blockSize >> > (chunk, d, dev_buf);
                //checkCUDAError("kernUpSweep failed!");
            }
            
            // Down Sweep
            cudaDeviceSynchronize();
            cudaMemset(dev_buf + chunk - 1, 0, sizeof(int)); // set root to zero
            //checkCUDAError("cudaMemset dev_buf[chunk-1] failed!");

            for (int d = power2-1; d >= 0; --d) {
                kernDownSweep <<<blocksPerGrid, blockSize>>> (chunk, d, dev_buf);
                //checkCUDAError("kernDownSweep failed!");
            }
            timer().endGpuTimer();

            cudaMemcpy(odata, dev_buf, arrSize, cudaMemcpyDeviceToHost);
            //checkCUDAError("cudaMemcpy odata failed!");
            cudaFree(dev_buf);
        }

        void scanOptimized(int n, int* odata, const int* idata) {
            int* dev_buf;

            int power2 = ilog2ceil(n);
            int chunk = 1 << power2;

            dim3 blocksPerGrid((chunk + blockSize - 1) / blockSize);
            size_t arrSize = n * sizeof(int);

            cudaMalloc((void**)&dev_buf, chunk * sizeof(int));
            //checkCUDAError("cudaMalloc dev_buf failed!");

            cudaMemcpy(dev_buf, idata, arrSize, cudaMemcpyHostToDevice);
            //checkCUDAError("cudaMemcpy idata to dev_buf failed!");
            if (chunk > n) {
                cudaMemset(dev_buf + n, 0, (chunk - n) * sizeof(int));
                //checkCUDAError("cudaMemset dev_buf[n] failed!");
            }

            timer().startGpuTimer();
            // Up Sweep
            int stride = chunk >> 1;
            for (int d = 0; d <= power2 - 1; ++d) {
                kernUpSweepOptimized << <blocksPerGrid, blockSize >> > (chunk, d, stride, dev_buf);
                stride >>= 1;
            }

            // Down Sweep
            cudaDeviceSynchronize();
            cudaMemset(dev_buf + chunk - 1, 0, sizeof(int)); // set root to zero
            //checkCUDAError("cudaMemset dev_buf[chunk-1] failed!");

            stride = chunk >> 1;
            for (int d = 1; d < chunk; d <<= 1) {
                kernDownSweepOptimized << <blocksPerGrid, blockSize >> > (chunk, d, stride, dev_buf);
                stride >>= 1;
            }
            timer().endGpuTimer();

            cudaMemcpy(odata, dev_buf, arrSize, cudaMemcpyDeviceToHost);
            //checkCUDAError("cudaMemcpy odata failed!");
            cudaFree(dev_buf);
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
            int* dev_odata;
            int* dev_bools;
            int* dev_indices;
            int count = 0;

            int power2 = ilog2ceil(n);
            int chunk = 1 << power2;

            dim3 chunkBlocksPerGrid((chunk + blockSize - 1) / blockSize);
            dim3 fullBlocksPerGrid((n + blockSize - 1) / blockSize);

            size_t arrSize = n * sizeof(int);
            cudaMalloc((void**)&dev_idata, arrSize);
            cudaMalloc((void**)&dev_odata, arrSize);
            cudaMalloc((void**)&dev_bools, arrSize);
            cudaMalloc((void**)&dev_indices, chunk * sizeof(int));

            // copy original data to GPU
            cudaMemcpy(dev_idata, idata, arrSize, cudaMemcpyHostToDevice);

            timer().startGpuTimer();
            // Step 1: Map
            Common::kernMapToBoolean << < fullBlocksPerGrid, blockSize >> > (n, dev_bools, dev_idata);

            // Step 2: Scan
            // copy bool array to index array for in-place operation, still need original bool array later for scatter
            cudaMemcpy(dev_indices, dev_bools, arrSize, cudaMemcpyDeviceToDevice);
            if (chunk > n) {
                cudaMemset(dev_indices + n, 0, (chunk - n) * sizeof(int));
            }

            // Up Sweep
            for (int d = 0; d <= power2 - 1; ++d) {
                kernUpSweep << <chunkBlocksPerGrid, blockSize >> > (chunk, d, dev_indices);
            }

            // set root to zero
            cudaMemset(dev_indices + chunk - 1, 0, sizeof(int)); 
            
            // Down Sweep
            for (int d = power2 - 1; d >= 0; --d) {
                kernDownSweep << <chunkBlocksPerGrid, blockSize >> > (chunk, d, dev_indices);
            }

            // Step 3: Scatter
            Common::kernScatter << <fullBlocksPerGrid, blockSize >> > (n,
                    dev_odata, dev_idata, dev_bools, dev_indices);
            timer().endGpuTimer();
            
            // Copy over last elements from GPU to local to return
            cudaMemcpy(&count, dev_indices + (n - 1), sizeof(int), cudaMemcpyDeviceToHost);
            count += (int)(idata[n - 1] != 0);
            cudaMemcpy(odata, dev_odata, arrSize, cudaMemcpyDeviceToHost);

            // free memory
            cudaFree(dev_idata);
            cudaFree(dev_odata);
            cudaFree(dev_bools);
            cudaFree(dev_indices);

            return count;
        }
    }
}

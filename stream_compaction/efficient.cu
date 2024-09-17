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

        __global__ void kernUpSweep(int offset, int offset2, int n, int* data) {
            int k = threadIdx.x + blockIdx.x * blockDim.x;
            if (k >= n) return;
            if (k % offset2 == 0) {
                data[k + offset2 - 1] += data[k + offset - 1];
            }
        }

        __global__ void kernDownSweep(int offset, int offset2, int n, int* data) {
            int k = threadIdx.x + blockIdx.x * blockDim.x;
            if (k >= n) return;
            if (k % offset2 == 0) {
                int t = data[k + offset - 1];                  
                data[k + offset - 1] = data[k + offset2 - 1]; 
                data[k + offset2 - 1] += t;                   
            }
        }
        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int* odata, const int* idata) {
            int power2 = ilog2ceil(n);
            int nonPowSize = 1 << power2;
            int* dev_data;

            cudaMalloc((void**)&dev_data, nonPowSize * sizeof(int));
            checkCUDAError("cudaMalloc dev_data failed!");
            cudaMemcpy(dev_data, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            checkCUDAError("cudaMemcpy idata failed!");
            //Fill the rest of the array with 0
            if (nonPowSize > n) {
                cudaMemset(&dev_data[n], 0, (nonPowSize - n) * sizeof(int));
                checkCUDAError("cudaMemset dev_data failed!");
            }

            int blockSize = 256;
            dim3 fullBlocksPerGrid((nonPowSize + blockSize - 1) / blockSize);

            timer().startGpuTimer();
            for (int d = 0; d <= power2 - 1; ++d) {
                int offset = 1 << d;
                int offset2 = 1 << (d + 1);
                kernUpSweep << <fullBlocksPerGrid, blockSize >> > (offset, offset2, nonPowSize, dev_data);
                checkCUDAError("kernUpSweep failed!");
            }
            cudaDeviceSynchronize();

            cudaMemset(&dev_data[nonPowSize - 1], 0, sizeof(int));
            checkCUDAError("cudaMemset dev_data failed!");

            for (int d = power2 - 1; d >= 0; --d) {
                int offset = 1 << d;
                int offset2 = 1 << (d + 1);
                kernDownSweep << <fullBlocksPerGrid, blockSize >> > (offset, offset2, nonPowSize, dev_data);
                checkCUDAError("kernDownSweep failed!");
            }

            cudaDeviceSynchronize();
            timer().endGpuTimer();

            cudaMemcpy(odata, dev_data, n * sizeof(int), cudaMemcpyDeviceToHost);
            checkCUDAError("cudaMemcpy odata failed!");

            cudaFree(dev_data);
        }


        /**f
        *
        *
        * Optimized version of scan
        *
        *
        **/
        __global__ void kernUpSweepOptimized(int offset, int offset2, int n, int* data) {
            int k = threadIdx.x + blockIdx.x * blockDim.x;
            // if n = 8, activeThreads: 8/2^1 = 4, 8/2^2 = 2, 8/2^3 = 1
            if (k >= n) return;
            //int offset = 1 << (d+1); offset = 2, 4, 8...
            //[0,1,2,3]*2 -> [0,2,4,6]; [0,1]*4 -> [0,4]; [0]*8 -> [0]
            k *= offset2;
            data[k+ offset2 - 1] += data[k + offset - 1];          
        }

        __global__ void kernDownSweepOptimized(int offset, int offset2, int n, int* data) {
            int k = threadIdx.x + blockIdx.x * blockDim.x;

            if (k >= n) return;
            k *= offset2;
            int t = data[k + offset - 1];
            data[k + offset - 1] = data[k + offset2 - 1];
            data[k + offset2 - 1] += t;
        }

        void scanOptimized(int n, int* odata, const int* idata) {
            int power2 = ilog2ceil(n);
            int nonPowSize = 1 << power2;
            int* dev_data;

            cudaMalloc((void**)&dev_data, nonPowSize * sizeof(int));
            checkCUDAError("cudaMalloc dev_data failed!");
            cudaMemcpy(dev_data, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            checkCUDAError("cudaMemcpy idata failed!");
            //Fill the rest of the array with 0
            if (nonPowSize > n) {
                cudaMemset(&dev_data[n], 0, (nonPowSize - n) * sizeof(int));
                checkCUDAError("cudaMemset dev_data failed!");
            }

            int blockSize = 256;
            dim3 fullBlocksPerGrid((nonPowSize + blockSize - 1) / blockSize);
            int sharedMemSize = blockSize * sizeof(int) * 2;
            timer().startGpuTimer();
            //UpSweep
            for (int d = 0; d <= power2 - 1; ++d) {
                int offset = 1 << d;
                int offset2 = 1 << (d + 1);
                // if n = 8, activeThreads: 8/2^1 = 4, 8/2^2 = 2, 8/2^3 = 1
                int activeThreads = nonPowSize >> (d + 1);
                dim3 dynamicBlocksPerGrid((activeThreads + blockSize - 1) / blockSize);
                kernUpSweepOptimized << <dynamicBlocksPerGrid, blockSize >> > (offset, offset2, activeThreads, dev_data);
                checkCUDAError("kernUpSweep failed!");
            }
            cudaDeviceSynchronize();

            cudaMemset(&dev_data[nonPowSize - 1], 0, sizeof(int));
            checkCUDAError("cudaMemset dev_data failed!");

            //DownSweep
            int count = 0;
            for (int d = power2 - 1; d >= 0; --d) {
                count++;
                int offset = 1 << d;
                int offset2 = 1 << (d + 1);
                int activeThreads = nonPowSize >> (d+1);
                dim3 dynamicBlocksPerGrid((activeThreads + blockSize - 1) / blockSize);
                kernDownSweepOptimized << <dynamicBlocksPerGrid, blockSize >> > (offset, offset2, activeThreads, dev_data);
                checkCUDAError("kernDownSweep failed!");
            }

            cudaDeviceSynchronize();
            timer().endGpuTimer();

            cudaMemcpy(odata, dev_data, n * sizeof(int), cudaMemcpyDeviceToHost);
            checkCUDAError("cudaMemcpy odata failed!");

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
            int power2 = ilog2ceil(n);
            int nonPowSize = 1 << power2;
            int* dev_idata;
            int* dev_bools;
            int* dev_indices;
            int* dev_odata;


            cudaMalloc((void**)&dev_idata, nonPowSize * sizeof(int));
            checkCUDAError("cudaMalloc dev_idata failed!");
            cudaMalloc((void**)&dev_bools, nonPowSize * sizeof(int));
            checkCUDAError("cudaMalloc dev_bools failed!");
            cudaMalloc((void**)&dev_indices, nonPowSize * sizeof(int));
            checkCUDAError("cudaMalloc dev_indices failed!");
            cudaMalloc((void**)&dev_odata, nonPowSize * sizeof(int));
            checkCUDAError("cudaMalloc dev_odata failed!");

            cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            checkCUDAError("cudaMemcpy idata failed!");
            if (nonPowSize > n) {
                cudaMemset(&dev_idata[n], 0, (nonPowSize - n) * sizeof(int));
                checkCUDAError("cudaMemset dev_data failed!");
            }

            int blockSize = 256;
            dim3 fullBlocksPerGrid((n + blockSize - 1) / blockSize);
            timer().startGpuTimer();

            //Step1: Compute temporary array containing and map to boolean
            StreamCompaction::Common::kernMapToBoolean << <fullBlocksPerGrid, blockSize >> > (nonPowSize, dev_bools, dev_idata);
            checkCUDAError("kernMapToBoolean failed!");

            //Step2: Scan the boolean array
            cudaMemcpy(dev_indices, dev_bools, nonPowSize * sizeof(int), cudaMemcpyDeviceToDevice);
            for (int d = 0; d <= power2 - 1; ++d) {
                int offset = 1 << d;
                int offset2 = 1 << (d + 1);
                kernUpSweep << <fullBlocksPerGrid, blockSize >> > (offset, offset2, nonPowSize, dev_indices);
                checkCUDAError("kernUpSweep failed!");
            }
            cudaDeviceSynchronize();

            cudaMemset(&dev_indices[nonPowSize - 1], 0, sizeof(int));
            checkCUDAError("cudaMemset dev_data failed!");
            cudaDeviceSynchronize();

            for (int d = power2 - 1; d >= 0; --d) {
                int offset = 1 << d;
                int offset2 = 1 << (d + 1);
                kernDownSweep << <fullBlocksPerGrid, blockSize >> > (offset, offset2, nonPowSize, dev_indices);
                checkCUDAError("kernDownSweep failed!");
            }

            //Step3: Scatter
            StreamCompaction::Common::kernScatter << <fullBlocksPerGrid, blockSize >> > (nonPowSize, dev_odata, dev_idata, dev_bools, dev_indices);
            checkCUDAError("kernScatter failed!");
            timer().endGpuTimer();

            //Copy the result to odata
            int count;
            cudaMemcpy(&count, &dev_indices[n - 1], sizeof(int), cudaMemcpyDeviceToHost);
            if (idata[n - 1] != 0) count++;
            cudaMemcpy(odata, dev_odata, n * sizeof(int), cudaMemcpyDeviceToHost);
            return count;
        }
    }
}

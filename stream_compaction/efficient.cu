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


        __global__ void upsweep(int* data, int twod, int twod1, int n) {
            int index = threadIdx.x + blockIdx.x * blockDim.x;
            int k = index * twod1;
            if (k + twod1 - 1 < n) {
                data[k + twod1 - 1] += data[k + twod - 1];
            }
        }


        __global__ void downsweep(int* data, int twod, int twod1, int n) {
            int index = threadIdx.x + blockIdx.x * blockDim.x;
            int k = index * twod1;
            if (k + twod1 - 1 < n) {
                int t = data[k + twod - 1];
                data[k + twod - 1] = data[k + twod1 - 1];
                data[k + twod1 - 1] += t;
            }
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */

         // Internal version that controls timing
        void scan_internal(int n, int* odata, const int* idata, bool timing) {
            if (timing) timer().startGpuTimer();

            int pow2Len = 1 << ilog2ceil(n);
            int* dev_data;
            cudaMalloc(&dev_data, pow2Len * sizeof(int));
            cudaMemset(dev_data, 0, pow2Len * sizeof(int));
            cudaMemcpy(dev_data, idata, n * sizeof(int), cudaMemcpyHostToDevice);

            int blockSize = 512;
            for (int d = 1; d <= ilog2ceil(pow2Len); ++d) {
                int twod = 1 << (d - 1);
                int twod1 = 1 << d;
                int numThreads = pow2Len / twod1;
                int numBlocks = (numThreads + blockSize - 1) / blockSize;
                upsweep << <numBlocks, blockSize >> > (dev_data, twod, twod1, pow2Len);
                cudaDeviceSynchronize();
            }

            cudaMemset(dev_data + pow2Len - 1, 0, sizeof(int)); // zero for exclusive

            for (int d = ilog2ceil(pow2Len); d >= 1; --d) {
                int twod = 1 << (d - 1);
                int twod1 = 1 << d;
                int numThreads = pow2Len / twod1;
                int numBlocks = (numThreads + blockSize - 1) / blockSize;
                downsweep << <numBlocks, blockSize >> > (dev_data, twod, twod1, pow2Len);
                cudaDeviceSynchronize();
            }

            cudaMemcpy(odata, dev_data, n * sizeof(int), cudaMemcpyDeviceToHost);
            cudaFree(dev_data);

            if (timing) timer().endGpuTimer();
        }

        void scan(int n, int* odata, const int* idata) {
            scan_internal(n, odata, idata, true);
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
            int* dev_idata, * dev_bools, * dev_indices, * dev_odata;

            int pow2n = 1 << ilog2ceil(n);
            cudaMalloc(&dev_idata, pow2n * sizeof(int));
            cudaMalloc(&dev_bools, pow2n * sizeof(int));
            cudaMalloc(&dev_indices, pow2n * sizeof(int));
            cudaMalloc(&dev_odata, pow2n * sizeof(int));
            cudaMemset(dev_odata, 0, pow2n * sizeof(int));

            cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);

            timer().startGpuTimer();

            int blockSize = 128;
            int numBlocks = (n + blockSize - 1) / blockSize;

            StreamCompaction::Common::kernMapToBoolean<<<numBlocks, blockSize >>>(
                n, dev_bools, dev_idata);
            cudaDeviceSynchronize();

            scan_internal(n, dev_indices, dev_bools, false);
            cudaDeviceSynchronize();

            StreamCompaction::Common::kernScatter<<<numBlocks, blockSize >>> (
                n, dev_odata, dev_idata, dev_bools, dev_indices);
            cudaDeviceSynchronize();

            timer().endGpuTimer();

            int count;
            cudaMemcpy(&count, dev_indices + n - 1, sizeof(int), cudaMemcpyDeviceToHost);

            int lastBool;
            cudaMemcpy(&lastBool, dev_bools + n - 1, sizeof(int), cudaMemcpyDeviceToHost);
            count += lastBool;

            cudaMemcpy(odata, dev_odata, count * sizeof(int), cudaMemcpyDeviceToHost);

            cudaFree(dev_idata);
            cudaFree(dev_bools);
            cudaFree(dev_indices);
            cudaFree(dev_odata);

            return count;
        }

    }
}
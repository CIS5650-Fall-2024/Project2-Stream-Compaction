#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

#define blockSize 128 

namespace StreamCompaction {
    namespace Efficient {
        using StreamCompaction::Common::PerformanceTimer;

        // Initialise device variables to use in compact()
        int *dev_bools, *dev_idata, *dev_odata, *dev_scanResult;

        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        // n & (n - 1): This expression clears the lowest set bit of n. 
        // If n is a power of 2, it has exactly one bit set, and subtracting 1 from it flips 
        // all the bits after the most significant bit. Performing n & (n - 1) results in 
        // 0 if and only if n is a power of 2.
        // n > 0: This ensures that n is positive, since negative numbers and zero are not powers of 2.
        __device__ __host__ bool isPowerOf2(int n) {
            return (n > 0) && ((n & (n - 1)) == 0);
        }

        __global__ void upsweep(int n, int d, int *data) {
            int k = threadIdx.x + (blockIdx.x * blockDim.x);

            if (k >= n) {
                return;
            }

            // is that shifting bits to the left by n positions is equivalent to multiplying the number by 2^n.
            // faster than calling pow(2, n)
            int two_pow_d_plus_1 = 1 << (d + 1);

            if (k % two_pow_d_plus_1 != 0) {
                return;
            }

            int two_pow_d = 1 << d;
            data[k + two_pow_d_plus_1 - 1] += data[k + two_pow_d - 1];
        }

        __global__ void init_downsweep(int n, int *odata) {
            int i = threadIdx.x + (blockIdx.x * blockDim.x);

            if (i == n - 1) {
                odata[i] = 0;
            } 
        }

        __global__ void downsweep(int n, int d, int *data) {
            int k = threadIdx.x + (blockIdx.x * blockDim.x);

            if (k >= n) {
                return;
            }

            // is that shifting bits to the left by n positions is equivalent to multiplying the number by 2^n.
            // faster than calling pow(2, n)
            int two_pow_d_plus_1 = 1 << (d + 1);

            if (k % two_pow_d_plus_1 != 0) {
                return;
            }

            int two_pow_d = 1 << d;
            int t = data[k + two_pow_d - 1];
            data[k + two_pow_d - 1] = data[k + two_pow_d_plus_1 - 1];
            data[k + two_pow_d_plus_1 - 1] += t;
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            timer().startGpuTimer();
            int *dev_odata_local;
            // Your intermediate array sizes will need to be rounded to the next power of two.
            int rounded_n = isPowerOf2(n) ? n : 1 << ilog2ceil(n);
   
            cudaMalloc((void**)&dev_odata_local, rounded_n * sizeof(int));

            // Copy idata to dev_odata_local first
            // Although we might initialise dev_odata_local with more than n elements, idata only contains n elements
            cudaMemcpy(dev_odata_local, idata, n * sizeof(int), cudaMemcpyHostToDevice);

            // Set up the grid and block sizes
            dim3 fullBlocksPerGrid((rounded_n + blockSize - 1) / blockSize);

            // upsweep
            for (int d = 0; d <= ilog2ceil(rounded_n) - 1; d++) {
                upsweep << <fullBlocksPerGrid, blockSize >> > (rounded_n, d, dev_odata_local);
            }

            // downsweep
            init_downsweep << <fullBlocksPerGrid, blockSize >> > (rounded_n, dev_odata_local);

            for (int d = ilog2ceil(rounded_n) - 1; d >= 0; d--) {
                downsweep << <fullBlocksPerGrid, blockSize >> > (rounded_n, d, dev_odata_local);
            }

            // Copy the result back to the host
            // Note that odata is only supposed to have n elements
            cudaMemcpy(odata, dev_odata_local, n * sizeof(int), cudaMemcpyDeviceToHost);

            cudaFree(dev_odata_local);

            timer().endGpuTimer();
        }

        /************************************************************************************
         * Define another scan function so that it can be called from compact()
         * Here dev_bools has already been initialised in compact()
         ************************************************************************************/
        void scan_without_timer(int n, int *odata, const int *idata) {
            // Initialise dev_scanResult. dev_bools only has n elements.
            cudaMemcpy(odata, idata, n * sizeof(int), cudaMemcpyDeviceToDevice);

            // Your intermediate array sizes will need to be rounded to the next power of two.
            int rounded_n = isPowerOf2(n) ? n : 1 << ilog2ceil(n);

            // Set up the grid and block sizes
            dim3 fullBlocksPerGrid((rounded_n + blockSize - 1) / blockSize);

            // upsweep
            for (int d = 0; d <= ilog2ceil(rounded_n) - 1; d++) {
                upsweep << <fullBlocksPerGrid, blockSize >> > (rounded_n, d, odata);
            }

            // downsweep
            init_downsweep << <fullBlocksPerGrid, blockSize >> > (rounded_n, odata);

            for (int d = ilog2ceil(rounded_n) - 1; d >= 0; d--) {
                downsweep << <fullBlocksPerGrid, blockSize >> > (rounded_n, d, odata);
            }
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

            // Initialise host variables for returning
            int *scanResult = new int[n];
            
            // Allocate memory on the device
            cudaMalloc((void**)&dev_idata, n * sizeof(int));
            cudaMalloc((void**)&dev_bools, n * sizeof(int));
            cudaMalloc((void**)&dev_scanResult, n * sizeof(int));
            cudaMalloc((void**)&dev_odata, n * sizeof(int));

            cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);

            // Set up the grid and block sizes
            dim3 fullBlocksPerGrid((n + blockSize - 1) / blockSize);

            // Map idata to a 0/1 array
            Common::kernMapToBoolean << <fullBlocksPerGrid, blockSize >> > (n, dev_bools, dev_idata);

            // Scan the boolean array
            scan_without_timer(n, dev_scanResult, dev_bools); // n will be rounded in the scan function

            // Perform scatter
            Common::kernScatter << <fullBlocksPerGrid, blockSize >> > (n, dev_odata, dev_idata, dev_bools, dev_scanResult);

            cudaMemcpy(scanResult, dev_scanResult, n * sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(odata, dev_odata, n * sizeof(int), cudaMemcpyDeviceToHost);
            int count = n == 0 ? 0 : scanResult[n - 1] + (idata[n - 1] != 0 ? 1 : 0);

            // Clean up device memory
            cudaFree(dev_idata);
            cudaFree(dev_bools);
            cudaFree(dev_scanResult);
            cudaFree(dev_odata);

            delete[] scanResult;

            timer().endGpuTimer();

            return count;
        }
    }
}

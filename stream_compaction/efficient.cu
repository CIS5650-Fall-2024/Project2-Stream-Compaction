#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

#define blockSize 128 // Default is 128

namespace StreamCompaction {
    namespace Efficient {
        using StreamCompaction::Common::PerformanceTimer;
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
            int *dev_odata;
            // Your intermediate array sizes will need to be rounded to the next power of two.
            int rounded_n = isPowerOf2(n) ? n : 1 << ilog2ceil(n);
   
            cudaMalloc((void**)&dev_odata, rounded_n * sizeof(int));

            // Copy idata to odata first
            cudaMemcpy(dev_odata, idata, rounded_n * sizeof(int), cudaMemcpyHostToDevice);

            // Set up the grid and block sizes
            dim3 fullBlocksPerGrid((rounded_n + blockSize - 1) / blockSize);

            // upsweep
            for (int d = 0; d <= ilog2ceil(rounded_n) - 1; d++) {
                upsweep << <fullBlocksPerGrid, blockSize >> > (rounded_n, d, dev_odata);
            }

            // downsweep
            init_downsweep << <fullBlocksPerGrid, blockSize >> > (rounded_n, dev_odata);

            for (int d = ilog2ceil(n) - 1; d >= 0; d--) {
                downsweep << <fullBlocksPerGrid, blockSize >> > (rounded_n, d, dev_odata);
            }

            // Copy the result back to the host
            cudaMemcpy(odata, dev_odata, rounded_n * sizeof(int), cudaMemcpyDeviceToHost);

            cudaFree(dev_odata);

            timer().endGpuTimer();
        }

        // __global__ void scan_without_timer(int n, int *odata, const int *idata){
        //     // Your intermediate array sizes will need to be rounded to the next power of two.
        //     int rounded_n = isPowerOf2(n) ? n : 1 << ilog2ceil(n);

        //     for (int d = 0; d <= ilog2ceil(rounded_n) - 1; d++) {   
        //         upsweep(rounded_n, d, odata);
        //     }
        // }
 
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
            // Initialise host variables
            int *bools = new int[n]; 
            int *scanResult = new int[n]; // Array to hold the scan result
            // Initialise device variables
            int *dev_bools, *dev_idata, *dev_odata, *dev_scanResult;
            // Allocate memory on the device
            cudaMalloc((void**)&dev_bools, n * sizeof(int));
            cudaMalloc((void**)&dev_idata, n * sizeof(int));
            cudaMalloc((void**)&dev_odata, n * sizeof(int));
            cudaMalloc((void**)&dev_scanResult, n * sizeof(int));

            cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);

            // Set up the grid and block sizes
            dim3 fullBlocksPerGrid((n + blockSize - 1) / blockSize);

            // Map idata to a 0/1 array
            Common::kernMapToBoolean << <fullBlocksPerGrid, blockSize >> > (n, dev_bools, dev_idata);
            cudaMemcpy(bools, dev_bools, n * sizeof(int), cudaMemcpyDeviceToHost);

            // Scan the boolean array
            scan(n, scanResult, bools); // The scan function will handle the rounding of n
            cudaMemcpy(dev_scanResult, scanResult, n * sizeof(int), cudaMemcpyHostToDevice);

            // Perform scatter
            Common::kernScatter << <fullBlocksPerGrid, blockSize >> > (n, dev_odata, dev_idata, dev_bools, dev_scanResult);

            // Clean up device memory
            cudaFree(dev_bools);
            cudaFree(dev_idata);
            cudaFree(dev_odata);
            cudaFree(dev_scanResult);

            timer().endGpuTimer();
            return -1;
        }
    }
}

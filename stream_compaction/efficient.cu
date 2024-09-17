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

        int nextPowerOf2(int n) {
            n--;
            n |= n >> 1;
            n |= n >> 2;
            n |= n >> 4;
            n |= n >> 8;
            n |= n >> 16;
            return n + 1;
        }

        __global__ void scanUpSweep(int* data, int n, int stride) {
            int idx = threadIdx.x + blockIdx.x * blockDim.x;
            int offset = stride * (2 * idx + 1) - 1;
            if (offset < n && offset + stride < n) {
                data[offset + stride] += data[offset];
            }
        }

        __global__ void scanDownSweep(int* data, int n, int stride) {
            int idx = threadIdx.x + blockIdx.x * blockDim.x;
            int offset = stride * (2 * idx + 1) - 1;
            if (offset + stride < n) {
                int temp = data[offset];
                data[offset] = data[offset + stride];
                data[offset + stride] += temp;
            }
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int* odata, const int* idata) {
            timer().startGpuTimer();

            int powerOf2 = nextPowerOf2(n);
            int* d_data;
            cudaMalloc(&d_data, powerOf2 * sizeof(int));
            cudaMemcpy(d_data, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            if (n < powerOf2) {
                cudaMemset(d_data + n, 0, (powerOf2 - n) * sizeof(int));
            }

            int threads = 256;
            int blocks = (powerOf2 + threads - 1) / threads;

            // Up-sweep
            for (int stride = 1; stride < powerOf2; stride *= 2) {
                scanUpSweep << <blocks, threads >> > (d_data, powerOf2, stride);
                cudaDeviceSynchronize();
            }

            // Set last element to 0 (for exclusive scan)
            int lastElement;
            cudaMemcpy(&lastElement, d_data + powerOf2 - 1, sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemset(d_data + powerOf2 - 1, 0, sizeof(int));

            // Down-sweep
            for (int stride = powerOf2 / 2; stride > 0; stride /= 2) {
                scanDownSweep << <blocks, threads >> > (d_data, powerOf2, stride);
                cudaDeviceSynchronize();
            }

            cudaMemcpy(odata, d_data, n * sizeof(int), cudaMemcpyDeviceToHost);
            cudaFree(d_data);

            timer().endGpuTimer();
        }

        __global__ void kernelPrefixSum(int* data, int n, int stride) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < n && idx >= stride) {
                data[idx] += data[idx - stride];
            }
        }

        __global__ void kernelCompact(int* odata, const int* idata, const int* flags, int n) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < n && idata[idx] != 0) {
                int writeIdx = (idx == 0) ? 0 : flags[idx - 1];
                odata[writeIdx] = idata[idx];
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
        int compact(int n, int* odata, const int* idata) {
            
            int* d_idata, * d_odata, * d_flags;
            cudaMalloc(&d_idata, n * sizeof(int));
            cudaMalloc(&d_odata, n * sizeof(int));
            cudaMalloc(&d_flags, n * sizeof(int));

            cudaMemcpy(d_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);

            int threads = 256;
            int blocks = (n + threads - 1) / threads;

            timer().startGpuTimer();
            // Map to flags (1 for non-zero, 0 for zero)
            Common::kernMapToBoolean << <blocks, threads >> > (n, d_flags, d_idata);
            cudaDeviceSynchronize();

            // Perform a parallel prefix sum on the flags
            for (int stride = 1; stride < n; stride *= 2) {
                kernelPrefixSum << <blocks, threads >> > (d_flags, n, stride);
                cudaDeviceSynchronize();
            }

            // Compact the array
            kernelCompact << <blocks, threads >> > (d_odata, d_idata, d_flags, n);
            cudaDeviceSynchronize();

            // Get the total number of non-zero elements
            int totalNonZero;
            cudaMemcpy(&totalNonZero, d_flags + n - 1, sizeof(int), cudaMemcpyDeviceToHost);

            // Copy the result back to host
            cudaMemcpy(odata, d_odata, totalNonZero * sizeof(int), cudaMemcpyDeviceToHost);

            cudaFree(d_idata);
            cudaFree(d_odata);
            cudaFree(d_flags);

            timer().endGpuTimer();
            return totalNonZero;
        }
    }
}

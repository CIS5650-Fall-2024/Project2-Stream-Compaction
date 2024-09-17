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

        // Finds the modulo of value with 2^power)
        __device__ int modPowBase2(int value, int power)
        {
            int shift = (sizeof(int) * 8 - power);
            return (value << shift) >> shift;
        }

        // Performs parallel reduction
        __global__ void kernUpSweep(int n, int d, int *data)
        {
            int index = (blockIdx.x * blockDim.x) + threadIdx.x;
            if (index >= n) return;

            if (modPowBase2(index, d + 1)) return;

            data[index + (1 << (d + 1)) - 1] += data[index + (1 << d) - 1];
        }

        // Build a scan from an up-swept input
        __global__ void kernDownSweep(int n, int d, int* data)
        {
            int index = (blockIdx.x * blockDim.x) + threadIdx.x;
            if (index >= n - 1) return;

            if (modPowBase2(index, d + 1)) return;

            int t = data[index + (1 << d) - 1];
            data[index + (1 << d) - 1] = data[index + (1 << (d + 1)) - 1];
            data[index + (1 << (d + 1)) - 1] += t;
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            
            // Number of levels needed for the scan
            int maxDepth = ilog2ceil(n);
            // Padded size of the work device array
            size_t dataSize = (1ull << maxDepth);
            // Reset value for down-sweeping
            int zero = 0;
            
            // Allocate device arrays
            int *dev_data;

            cudaMalloc((void**)&dev_data, dataSize * sizeof(int));
            checkCUDAError("cudaMalloc Efficient::scan::dev_data failed!");

            cudaMemcpy(dev_data, idata, n * sizeof(int), cudaMemcpyHostToDevice);

            dim3 gridDim((dataSize + blockSize - 1) / blockSize);

            timer().startGpuTimer();
            
            // Up-Sweep
            for (int d = 0; d < maxDepth; ++d)
            {
                kernUpSweep<<<gridDim, blockSize>>>(dataSize, d, dev_data);
            }

            // Replace the last element with 0
            cudaMemcpy(dev_data + dataSize - 1, &zero, sizeof(int), cudaMemcpyHostToDevice);

            // Down-Sweep
            for (int d = maxDepth - 1; d >= 0; --d)
            {
                kernDownSweep<<<gridDim, blockSize>>>(dataSize, d, dev_data);
            }

            timer().endGpuTimer();

            // Copy the output data
            cudaMemcpy(odata, dev_data, n * sizeof(int), cudaMemcpyDeviceToHost);

            // Free device arrays
            cudaFree(dev_data);
            checkCUDAError("cudaFree Efficient::scan failed!");
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
            // Number of levels needed for the scan
            int maxDepth = ilog2ceil(n);
            // Padded size of the work device array
            size_t dataSize = (1ull << maxDepth);
            // Reset value for down-sweeping
            int zero = 0;

            // Allocate device arrays
            int *dev_idata, *dev_bools, *dev_indices, *dev_odata;

            cudaMalloc((void**)&dev_idata, dataSize * sizeof(int));
            checkCUDAError("cudaMalloc Efficient::compact::dev_idata failed!");

            cudaMalloc((void**)&dev_bools, dataSize * sizeof(int));
            checkCUDAError("cudaMalloc Efficient::compact::dev_bools failed!");

            cudaMalloc((void**)&dev_indices, dataSize * sizeof(int));
            checkCUDAError("cudaMalloc Efficient::compact::dev_indices failed!");

            cudaMalloc((void**)&dev_odata, dataSize * sizeof(int));
            checkCUDAError("cudaMalloc Efficient::compact::dev_odata failed!");

            cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);

            dim3 gridDim((dataSize + blockSize - 1) / blockSize);

            timer().startGpuTimer();

            // Map to boolean
            Common::kernMapToBoolean<<<gridDim, blockSize>>>(dataSize, dev_bools, dev_idata);


            // Scan the indices
            cudaMemcpy(dev_indices, dev_bools, dataSize * sizeof(int), cudaMemcpyHostToHost);

            // Up-Sweep
            for (int d = 0; d < maxDepth; ++d)
            {
                kernUpSweep<<<gridDim, blockSize>>>(dataSize, d, dev_indices);
            }

            // Replace the last element with 0
            cudaMemcpy(dev_indices + dataSize - 1, &zero, sizeof(int), cudaMemcpyHostToDevice);

            // Down-Sweep
            for (int d = maxDepth - 1; d >= 0; --d)
            {
                kernDownSweep<<<gridDim, blockSize>>>(dataSize, d, dev_indices);
            }


            // Scatter
            Common::kernScatter<<<gridDim, blockSize>>>(dataSize, dev_odata, dev_idata, dev_bools, dev_indices);

            timer().endGpuTimer();

            // Retrieve the output size
            int* ptr_size = new int();
            int* ptr_doLast = new int();
            cudaMemcpy(ptr_size, dev_indices + dataSize - 1, sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(ptr_doLast, dev_bools + dataSize - 1, sizeof(int), cudaMemcpyDeviceToHost);

            int size = *ptr_size + (*ptr_doLast ? 1 : 0);

            delete(ptr_doLast);
            delete(ptr_size);

            // Copy the output data
            cudaMemcpy(odata, dev_odata, size * sizeof(int), cudaMemcpyDeviceToHost);

            // Free device arrays
            cudaFree(dev_odata);
            cudaFree(dev_indices);
            cudaFree(dev_bools);
            cudaFree(dev_idata);
            checkCUDAError("cudaFree Efficient::compact failed!");

            return size;
        }
    }
}

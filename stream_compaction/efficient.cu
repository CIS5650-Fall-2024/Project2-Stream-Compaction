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

        __global__ void kernUpSweep(int n, int *data, int step) {
            int index = (blockIdx.x * blockDim.x) + threadIdx.x;
            if (index < n && index % step == 0) {
                data[index + step - 1] += data[index + (step >> 1) - 1];
            }
        }

        __global__ void kernSetLastZero(int n, int *data) {
            int index = (blockIdx.x * blockDim.x) + threadIdx.x;
            if (index == n - 1) {
                data[index] = 0;
            }
        }

        __global__ void kernDownSweep(int n, int *data, int step) {
            int index = (blockIdx.x * blockDim.x) + threadIdx.x;
            if (index < n && index % step == 0) {
                int idx1 = index + (step >> 1) - 1, idx2 = index + step - 1;
                int t = data[idx1];
                data[idx1] = data[idx2];
                data[idx2] += t;
            }
        }

        __global__ void kernCount(int n, int *count, const int *idata, const int *indices) {
            int index = (blockIdx.x * blockDim.x) + threadIdx.x;
            if (index == n - 1) {
                count[0] = indices[index] + (idata[index] ? 1 : 0);
            }
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            int *dev_data;
            int rounds = ilog2ceil(n);
            int size = 1 << rounds;
            cudaMalloc((void**)&dev_data, size * sizeof(int));
            checkCUDAError("cudaMalloc dev_data failed!");
            cudaMemcpy(dev_data, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            dim3 blocks((n + blockSize - 1) / blockSize);
            
            timer().startGpuTimer();
            int step = 2;
            for (int i = 0; i < rounds; i++) {
                kernUpSweep<<<blocks, blockSize>>>(size, dev_data, step);
                checkCUDAError("kernUpSweep failed!");
                step <<= 1;
            }
            kernSetLastZero<<<blocks, blockSize>>>(size, dev_data);
            checkCUDAError("kernSetLastZero failed!");
            step = size;
            for (int i = 0; i < rounds; i++) {
                kernDownSweep<<<blocks, blockSize>>>(size, dev_data, step);
                checkCUDAError("kernDownSweep failed!");
                step >>= 1;
            }
            timer().endGpuTimer();
            cudaMemcpy(odata, dev_data, n * sizeof(int), cudaMemcpyDeviceToHost);
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
            int *dev_odata, *dev_idata, *dev_indices, *dev_count;
            int rounds = ilog2ceil(n);
            int size = 1 << rounds;
            cudaMalloc((void**)&dev_odata, n * sizeof(int));
            checkCUDAError("cudaMalloc dev_odata failed!");
            cudaMalloc((void**)&dev_idata, n * sizeof(int));
            checkCUDAError("cudaMalloc dev_idata failed!");
            cudaMalloc((void**)&dev_indices, size * sizeof(int));
            checkCUDAError("cudaMalloc dev_indices failed!");
            cudaMalloc((void**)&dev_count, sizeof(int));
            checkCUDAError("cudaMalloc dev_count failed!");
            cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            dim3 blocks((n + blockSize - 1) / blockSize);

            timer().startGpuTimer();
            StreamCompaction::Common::kernMapToBoolean<<<blocks, blockSize>>>(n, dev_indices, dev_idata);
            checkCUDAError("kernMapToBoolean failed!");
            int step = 2;
            for (int i = 0; i < rounds; i++) {
                kernUpSweep<<<blocks, blockSize>>>(size, dev_indices, step);
                checkCUDAError("kernUpSweep failed!");
                step <<= 1;
            }
            kernSetLastZero<<<blocks, blockSize>>>(size, dev_indices);
            checkCUDAError("kernSetLastZero failed!");
            step = size;
            for (int i = 0; i < rounds; i++) {
                kernDownSweep<<<blocks, blockSize>>>(size, dev_indices, step);
                checkCUDAError("kernDownSweep failed!");
                step >>= 1;
            }
            StreamCompaction::Common::kernScatter<<<blocks, blockSize>>>(n, dev_odata, dev_idata, dev_idata, dev_indices);
            kernCount<<<blocks, blockSize>>>(n, dev_count, dev_idata, dev_indices);
            checkCUDAError("kernCount failed!");
            timer().endGpuTimer();
            int count;
            cudaMemcpy(&count, dev_count, sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(odata, dev_odata, count * sizeof(int), cudaMemcpyDeviceToHost);
            cudaFree(dev_odata);
            cudaFree(dev_idata);
            cudaFree(dev_indices);
            cudaFree(dev_count);
            return count;
        }
    }
}

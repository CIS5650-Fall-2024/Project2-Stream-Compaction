#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"
#include <device_launch_parameters.h>

#define blockSize 1024

namespace StreamCompaction {
    namespace Efficient {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        __host__ __device__ int pow_two(int t) {
            return 1 << t;
        }

        __global__ void up_sweep(int n, int d, int* odata, const int* idata) {
            int index = (blockIdx.x * blockDim.x) + threadIdx.x;
            if (index < n) {
                odata[index] = idata[index];
                int step = pow_two(d + 1);
                if (index % step == 0) {
                    int t = pow_two(d);
                    odata[index + step - 1] = idata[index + step - 1] + idata[index + t - 1];
                }
            }
        }

        __global__ void down_sweep(int n, int d, int* odata, int* idata) {
            int index = (blockIdx.x * blockDim.x) + threadIdx.x;
            if (index < n) {
                int step = pow_two(d + 1);
                odata[index] = idata[index];
                if (index % step == 0) {
                    int t = pow_two(d);
                    odata[index + t - 1] = idata[index + step - 1];
                    odata[index + step - 1] += idata[index + t - 1];
                }
            }
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            int size = pow_two(ilog2ceil(n));
            int depth = ilog2ceil(size);
            dim3 fullBlockPerGrid = ((size + blockSize - 1) / blockSize);

            int* buffer1;
            int* buffer2;
            cudaMalloc((void**)&buffer1, size * sizeof(int));
            checkCUDAErrorFn("failed to allocate buffer1");

            cudaMalloc((void**)&buffer2, size * sizeof(int));
            checkCUDAErrorFn("failed to allocate buffer2");

            cudaMemcpy(buffer1, idata, n * sizeof(int), cudaMemcpyHostToDevice);

            timer().startGpuTimer();

            for (int d = 0; d < depth; d++) {
                up_sweep << <fullBlockPerGrid, blockSize >> > (size, d, buffer2, buffer1);
                std::swap(buffer1, buffer2);
            }

            cudaMemset(buffer1 + size - 1, 0, sizeof(int));

            for (int d = depth - 1; d >= 0; d--) {
                down_sweep << <fullBlockPerGrid, blockSize >> > (size, d, buffer2, buffer1);
                std::swap(buffer1, buffer2);
            }

            timer().endGpuTimer();

            cudaMemcpy(odata, buffer1, n * sizeof(int), cudaMemcpyDeviceToHost);

            cudaFree(buffer1);
            cudaFree(buffer2);
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
            // TODO
            int size = pow_two(ilog2ceil(n));
            int depth = ilog2ceil(size);
            dim3 fullBlockPerGrid = ((size + blockSize - 1) / blockSize);

            int* bools;
            int* indices;
            int* buffer1;
            int* buffer2;

            cudaMalloc((void**)&bools, size * sizeof(int));
            checkCUDAErrorFn("failed to allocate bools");

            cudaMalloc((void**)&indices, size * sizeof(int));
            checkCUDAErrorFn("failed to allocate indices");

            cudaMalloc((void**)&buffer1, size * sizeof(int));
            checkCUDAErrorFn("failed to allocate buffer1");

            cudaMalloc((void**)&buffer2, size * sizeof(int));
            checkCUDAErrorFn("failed to allocate buffer2");

            cudaMemcpy(buffer1, idata, n * sizeof(int), cudaMemcpyHostToDevice);

            Common::kernMapToBoolean << <fullBlockPerGrid, blockSize >> > (size, bools, buffer1);

            cudaMemcpy(indices, bools, n * sizeof(int), cudaMemcpyDeviceToDevice);

            for (int d = 0; d < depth; d++) {
                up_sweep << <fullBlockPerGrid, blockSize >> > (size, d, buffer2, indices);
                std::swap(indices, buffer2);
            }

            cudaMemset(indices + size - 1, 0, sizeof(int));

            for (int d = depth - 1; d >= 0; d--) {
                down_sweep << <fullBlockPerGrid, blockSize >> > (size, d, buffer2, indices);
                std::swap(indices, buffer2);
            }

            Common::kernScatter << <fullBlockPerGrid, blockSize >> > (size, buffer2, buffer1, bools, indices);

            int count = 0;
            cudaMemcpy(&count, indices + size - 1, sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(odata, buffer2, n * sizeof(int), cudaMemcpyDeviceToHost);

            cudaFree(bools);
            cudaFree(indices);
            cudaFree(buffer1);
            cudaFree(buffer2);

            timer().endGpuTimer();
            return count;
        }
    }
}

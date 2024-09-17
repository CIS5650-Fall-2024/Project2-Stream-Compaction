#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"
#include <device_launch_parameters.h>

#define blockSize 1024

namespace StreamCompaction {
    namespace Naive {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        __global__ void scan_one_iteration(int n, int d, int* odata, const int* idata) {
            int index = (blockIdx.x * blockDim.x) + threadIdx.x;
            if (index < n) {
                int pow_d = pow(2, d - 1);
                if (index >= pow_d) {
                    odata[index] = idata[index - pow_d] + idata[index];
                }
                else {
                    odata[index] = idata[index];
                }
            }
        }

        __global__ void shift_right(int n, int* odata, const int* idata) {
            int index = (blockIdx.x * blockDim.x) + threadIdx.x;
            if (index < n) {
                if (index == 0) {
                    odata[index] = 0;
                }
                else {
                    odata[index] = idata[index - 1];
                }
            }
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            int* buffer1;
            int* buffer2;
            cudaMalloc((void**)&buffer1, n * sizeof(int));
            checkCUDAErrorFn("failed to allocate buffer1");

            cudaMalloc((void**)&buffer2, n * sizeof(int));
            checkCUDAErrorFn("failed to allocate buffer2");

            cudaMemcpy(buffer1, idata, n * sizeof(int), cudaMemcpyHostToDevice);

            dim3 fullBlockPerGrid = ((n + blockSize - 1) / blockSize);
            int depth = ilog2ceil(n);

            timer().startGpuTimer();

            for (int d = 1; d <= depth; d++) {
                scan_one_iteration<<< fullBlockPerGrid, blockSize>>> (n, d, buffer2, buffer1);
                std::swap(buffer1, buffer2);
            }

            shift_right << < fullBlockPerGrid, blockSize >> > (n, buffer2, buffer1);

            timer().endGpuTimer();

            cudaMemcpy(odata, buffer2, n * sizeof(int), cudaMemcpyDeviceToHost);

            cudaFree(buffer1);
            cudaFree(buffer2);
        }
    }
}

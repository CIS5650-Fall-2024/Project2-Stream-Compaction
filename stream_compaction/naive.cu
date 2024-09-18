#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"
#include <device_launch_parameters.h>

namespace StreamCompaction {
    namespace Naive {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        __device__ int pow_two(int d) {
            return 1 << d;
        }

        __global__ void scan_one_iteration(int n, int d, int* odata, const int* idata) {
            int index = (blockIdx.x * blockDim.x) + threadIdx.x;
            if (index < n) {
                int pow_d = pow_two(d - 1);
                if (index >= pow_d) {
                    odata[index] = idata[index - pow_d] + idata[index];
                }
                else {
                    odata[index] = idata[index];
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

            odata[0] = 0;
            cudaMemcpy(odata + 1, buffer1, (n - 1) * sizeof(int), cudaMemcpyDeviceToHost);

            timer().endGpuTimer();

            cudaFree(buffer1);
            cudaFree(buffer2);
        }
    }
}

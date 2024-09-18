#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"
#include "device_launch_parameters.h"

namespace StreamCompaction {
    namespace Naive {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }
        // TODO: __global__
        __global__ void kernHandleNonPower(int n, int d, int* buffer) {
            int index = threadIdx.x + blockIdx.x * blockDim.x;
            int pow2tod = 1 << d;

            if (index >= n - pow2tod) return;

            buffer[pow2tod + index] += buffer[index];
        }

        __global__ void kernNaiveScanStep(int n, int d, const int* readBuffer, int* writeBuffer) {
            // compute thread index
            int index = threadIdx.x + blockIdx.x * blockDim.x;
            if (index >= n) return;

            int pow2tod = 1 << d;

            if (index > pow2tod) {
                writeBuffer[index] = readBuffer[index] + readBuffer[index - pow2tod];
            }
            else {
                writeBuffer[index] = readBuffer[index];
            }
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            int blockSize = 128;
            dim3 fullBlocksPerGrid{ (unsigned int) (n + blockSize - 1) / blockSize };

            int* dev_buffer1;
            int* dev_buffer2;

            cudaMalloc((void**)&dev_buffer1, n * sizeof(int));
            checkCUDAError("cudaMalloc dev_buffer1 failed!");
            cudaMalloc((void**)&dev_buffer2, n * sizeof(int));
            checkCUDAError("cudaMalloc dev_buffer2 failed!");

            cudaMemcpy(dev_buffer2, idata, n * sizeof(int), cudaMemcpyHostToDevice);

            timer().startGpuTimer();
            StreamCompaction::Common::shiftArrayElements<<<fullBlocksPerGrid, blockSize>>>(n, 1, dev_buffer2, dev_buffer1);
            checkCUDAError("shiftArrayElements failed!");
            cudaDeviceSynchronize();

            for (int d = 0; d < ilog2(n); ++d) {
                kernNaiveScanStep <<<fullBlocksPerGrid, blockSize>>>(n, d, dev_buffer1, dev_buffer2);
                checkCUDAError("naiveScanStep failed!");
                cudaDeviceSynchronize();

                std::swap(dev_buffer1, dev_buffer2);
            }
            // perform last step 
            if ((1 << ilog2(n)) != n) {
                fullBlocksPerGrid.x = (n - (1 << ilog2(n)) + blockSize - 1) / blockSize;
                kernHandleNonPower<<<fullBlocksPerGrid, blockSize>>>(n, ilog2(n), dev_buffer1);
                checkCUDAError("handleNonPower failed!");
                cudaDeviceSynchronize();
            }
            timer().endGpuTimer();

            cudaMemcpy(odata, dev_buffer1, n * sizeof(int), cudaMemcpyDeviceToHost);

            cudaFree(dev_buffer1);
            cudaFree(dev_buffer2);
        }
    }
}
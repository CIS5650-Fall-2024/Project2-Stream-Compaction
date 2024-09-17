#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"

namespace StreamCompaction {
    namespace Naive {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }
        // TODO: __global__
        __global__ void KernScanAtDepthD(int n, int d, int* iBuffer, int* oBuffer) {
            int index = (blockIdx.x * blockDim.x) + threadIdx.x;
            int offset = pow(2, d - 1);
            if (index < n) {
                if (index >= offset) {
                    oBuffer[index] = iBuffer[index] + iBuffer[index - offset];
                }
                else {
                    oBuffer[index] = iBuffer[index];
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
            cudaMalloc((void**)&buffer2, n * sizeof(int));
            cudaMemcpy(buffer2, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            timer().startGpuTimer();
            // TODO
            for (int d = 1; d <= ilog2ceil(n); d++) {
                dim3 fullBlocksPerGrid((n + blockSize - 1) / blockSize);
                int* iBuffer;
                int* oBuffer;
                if (d % 2 == 1) {
                    iBuffer = buffer2;
                    oBuffer = buffer1;
                }
                else {
                    iBuffer = buffer1;
                    oBuffer = buffer2;
                }
                KernScanAtDepthD<<<fullBlocksPerGrid, blockSize>>> (n, d, iBuffer, oBuffer);
                checkCUDAError("KernScanAtDepthD failed!");
            }

            timer().endGpuTimer();
            //Exclusive shift + copying correct buffer over based on parity
            odata[0] = 0;
            if (ilog2ceil(n) % 2 == 1) {
                cudaMemcpy(odata + 1, buffer1, (n - 1) * sizeof(int), cudaMemcpyDeviceToHost);
            }
            else {
                cudaMemcpy(odata + 1, buffer2, (n - 1) * sizeof(int), cudaMemcpyDeviceToHost);
            }
            cudaFree(buffer1);
            cudaFree(buffer2);
        }
    }
}

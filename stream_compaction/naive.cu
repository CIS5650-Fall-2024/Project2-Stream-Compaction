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

        __global__ void KernScanAtDepthD(int n, int offset, int* iBuffer, int* oBuffer) {
            int index = (blockIdx.x * blockDim.x) + threadIdx.x;
            if (index < n) {
                if (index >= offset) {
                    oBuffer[index] = iBuffer[index] + iBuffer[index - offset];
                }
                else {
                    oBuffer[index] = iBuffer[index];
                }
                
            }
        }

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
                KernScanAtDepthD<<<fullBlocksPerGrid, blockSize>>>(n, (1 << d - 1), buffer2, buffer1);
                if (d != ilog2ceil(n)) {
                    std::swap(buffer2, buffer1);
                }
                
                checkCUDAError("KernScanAtDepthD failed!");
            }

            timer().endGpuTimer();
            //Exclusive shift + copying correct buffer over based on parity
            odata[0] = 0;
            cudaMemcpy(odata + 1, buffer1, (n-1) * sizeof(int), cudaMemcpyDeviceToHost);
            cudaFree(buffer1);
            cudaFree(buffer2);
        }
    }
}

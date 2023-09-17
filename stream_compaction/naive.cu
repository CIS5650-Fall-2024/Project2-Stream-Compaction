#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"
#define BlockSize 256

namespace StreamCompaction {
    namespace Naive {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }
        // TODO: __global__
        __global__ void kernScan(int n, int* odata, const int* idata, int d) {
             int index = threadIdx.x + (blockIdx.x * blockDim.x);
            if (index >= n) {
                return;
            }
            if (odata[index] != idata[index]) {
                odata[index] = idata[index];
            }
            
            if (index >= (int)powf(2, d - 1)) {
                odata[index] = idata[index - (int)powf(2, d - 1)] + idata[index];
            }
        }

        __global__ void kernelIncToExc(const int n, int* odata, const int* idata) {
            int index = (blockIdx.x * blockDim.x) + threadIdx.x;
            if (index >= n) {
                return;
            }

            
            if (index == 0) {
                odata[0] = 0;
            }
            else {
                odata[index] = idata[index - 1];
            }
        }


        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            int* device_A;
            int* device_B;

  
            cudaMalloc((void**)&device_A, n * sizeof(int));
            checkCUDAError("cudaMalloc device_A failed!");
            cudaMalloc((void**)&device_B, n * sizeof(int));
            checkCUDAError("cudaMalloc device_B failed!");
            
            cudaMemcpy(device_A, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            checkCUDAError("cudaMemcpy cudaMemcpyHostToDevice device_A to idata failed!");

            dim3 blocksPerGrid((n + BlockSize - 1) / BlockSize);

            timer().startGpuTimer();
            // TODO
            int* temp;
            for (int d = 1; d <= ilog2ceil(n); d++) {
                kernScan << <blocksPerGrid, BlockSize >> > (n, device_B, device_A, d);
                temp = device_A;
                device_A = device_B;
                device_B = temp;
            }

            kernelIncToExc << <blocksPerGrid, BlockSize >> > (n, device_B, device_A);
            

            timer().endGpuTimer();

            cudaMemcpy(odata, device_B, n * sizeof(int), cudaMemcpyDeviceToHost);
            
            checkCUDAError("cudaMemcpy cudaMemcpyDeviceToHost odata to device_B failed!");

            cudaFree(device_A);
            checkCUDAError("cudaFree device_A failed!");
            cudaFree(device_B);
            checkCUDAError("cudaFree device_B failed!");

        }
    }
}

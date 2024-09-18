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

        // naive scan (kernel)
        __global__ void scanKernel(int* g_odata, const int* g_idata, int n, int offset) {
            // get index
            int index = threadIdx.x + blockIdx.x * blockDim.x;

            // return early if bad val
            if (index >= n) return;

            
            if (index >= offset) {
                g_odata[index] = g_idata[index] + g_idata[index - offset];

            } else {
                g_odata[index] = g_idata[index];
            }
        }
 

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int* odata, const int* idata) {
            // allocate memory on device
            int* d_ping, * d_pong;
            cudaMalloc((void**)&d_ping, n * sizeof(int));
            checkCUDAError("cudaMalloc d_ping");
            cudaMalloc((void**)&d_pong, n * sizeof(int));
            checkCUDAError("cudaMalloc d_pong");

            // copy data over
            cudaMemcpy(d_ping + 1, idata, (n - 1) * sizeof(int), cudaMemcpyHostToDevice);
            checkCUDAError("cudaMemcpy to d_ping");

            int zero = 0;
            cudaMemcpy(d_ping, &zero, sizeof(int), cudaMemcpyHostToDevice);
            checkCUDAError("cudaMemcpy identity to d_ping");

            // setup info
            int logn = ilog2ceil(n);
            int blockSize = 384;
            int numBlocks = (n + blockSize - 1) / blockSize;

            timer().startGpuTimer();

            // do scan on subset of array
            for (int d = 0; d < logn; d++) {
                int offset = 1 << d;

                // launch kernel
                scanKernel<<<numBlocks, blockSize>>>(d_pong, d_ping, n, offset);
                cudaDeviceSynchronize();
                checkCUDAError("scanKernel execution");

                // swap pointers
                int* temp = d_ping;
                d_ping = d_pong;
                d_pong = temp;
            }

            timer().endGpuTimer();

            // copy result back to host memory
            cudaMemcpy(odata, d_ping, n * sizeof(int), cudaMemcpyDeviceToHost);
            checkCUDAError("cudaMemcpy to odata");

            // free device memory
            cudaFree(d_ping);
            cudaFree(d_pong);
        }
    }
}

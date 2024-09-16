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
        __global__ void scan_kernel(int n, int* odata, const int* idata, int step) {
            int index = (blockIdx.x * blockDim.x) + threadIdx.x;

            if (index < n) {
                if (index >= step) {
                    odata[index] = idata[index] + idata[index - step];
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
            int* dev_buffer_A;
            int* dev_buffer_B;

            cudaMalloc((void**)&dev_buffer_A, n * sizeof(int));
            cudaMalloc((void**)&dev_buffer_B, n * sizeof(int));

            cudaMemcpy(dev_buffer_A, idata, n * sizeof(int), cudaMemcpyHostToDevice);

            int block_size = 128;
            dim3 fullBlocksPerGrid((n + block_size - 1) / block_size);

            timer().startGpuTimer();
            // TODO
            for (int d = 1; d <= ilog2ceil(n); d++) {
                int step = 1 << (d - 1);

                if ((d % 2) == 1) {
                    scan_kernel << <fullBlocksPerGrid, block_size >> > (n, dev_buffer_B
                        , dev_buffer_A, step);
                }
                else {
                    scan_kernel << <fullBlocksPerGrid, block_size >> > (n, dev_buffer_A
                        , dev_buffer_B, step);
                }

            }
            timer().endGpuTimer();
            if (ilog2ceil(n) % 2 == 1) {
                cudaMemcpy(odata, dev_buffer_B, n * sizeof(int), cudaMemcpyDeviceToHost);
            }
            else {
                cudaMemcpy(odata, dev_buffer_A, n * sizeof(int), cudaMemcpyDeviceToHost);
            }
            for (int i = n - 1; i > 0; i--) {
                odata[i] = odata[i - 1];
            }
            odata[0] = 0;

            cudaFree(dev_buffer_A);
            cudaFree(dev_buffer_B);
            
        }
    }
}

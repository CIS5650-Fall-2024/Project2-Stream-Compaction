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

        __global__ void kernScan(int n, int d, int* odata, const int* idata) {
            int k = (blockDim.x * blockIdx.x) + threadIdx.x;
            if (k >= n) {
                return;
            }
            int pow_res = (1 << d - 1);
            if (k >= pow_res) {
                int idx = k - pow_res;
                odata[k] = idata[idx] + idata[k];
            }
            else {
                odata[k] = idata[k];
            }
        }

        __global__ void kernMakeScanExclusive(int n, int* odata, int* idata) {
            int k = (blockDim.x * blockIdx.x) + threadIdx.x;
            if (k >= n) {
                return;
            }
            odata[k] = k == 0 ? 0 : idata[k - 1];
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {

            int* data_a;
            int* data_b;
            cudaMalloc((void**)&data_a, sizeof(int) * n);
            cudaMalloc((void**)&data_b, sizeof(int) * n);

            cudaMemcpy(data_a, idata, sizeof(int) * n, cudaMemcpyHostToDevice);

            timer().startGpuTimer();

            int log = ilog2ceil(n);

            int blockSize = 128;
            dim3 blockDim((n + blockSize - 1) / blockSize);

            for (int d = 1; d <= log; d++) {
                // k = n? does not seem necessary, can prob find a log val that we can limit k to. then do [k, n]
                kernScan << <blockDim, blockSize >> > (n, d, data_b, data_a);
                int* temp = data_a;
                data_a = data_b;
                data_b = temp;
            }

            kernMakeScanExclusive << <blockDim, blockSize >> > (n, data_b, data_a);

            timer().endGpuTimer();

            cudaMemcpy(odata, data_b, sizeof(int) * n, cudaMemcpyDeviceToHost);

            cudaFree(data_a);
            cudaFree(data_b);
        }
    }
}

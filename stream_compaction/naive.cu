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

        __global__ void kernScan(int n, int d, int* odata, const int* idata) {
            int k = (blockDim.x * blockIdx.x) + threadIdx.x;
            if (k > n) {
                return;
            }
            int pow_res = pow(2, d - 1);
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
            if (k > n) {
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

            cudaMemcpy(data_b, idata, sizeof(int) * n, cudaMemcpyHostToDevice);

            timer().startGpuTimer();

            int log = ilog2ceil(n);

            int blockSize = 128;
            dim3 blockDim((n + blockSize - 1) / blockSize);
            bool a_has_output = true;

            for (int d = 1; d <= log; d++) {
                // k = n? does not seem necessary, can prob find a log val that we can limit k to. then do [k, n]
                kernScan << <blockDim, blockSize >> > (n, d, data_a, data_b);
                int* temp = data_a;
                data_a = data_b;
                data_b = temp;
                a_has_output = !a_has_output;
            }

            int *out_arr, *in_arr;
            if (a_has_output) {
                out_arr = data_a;
                in_arr = data_b;
            }
            else {
                out_arr = data_b;
                in_arr = data_a;
            }
            kernMakeScanExclusive << <blockDim, blockSize >> > (n, out_arr, in_arr);

            timer().endGpuTimer();

            cudaMemcpy(odata, out_arr, sizeof(int) * n, cudaMemcpyDeviceToHost);

            cudaFree(data_a);
            cudaFree(data_b);
        }
    }
}

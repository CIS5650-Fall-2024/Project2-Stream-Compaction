#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"

#define BLOCK_SIZE 128
namespace StreamCompaction {
    namespace Naive {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }
        // TODO: __global__
        __global__ void kernNaiveScan(int N, int two_d_1, int* input, int* output)
        {
            int index = blockIdx.x * blockDim.x + threadIdx.x;
            if (index > N) return;
            output[index] = index >= two_d_1 ? input[index - two_d_1] + input[index] : input[index];
        }

        __global__ void kernNaiveShift(int N, int* input, int* output)
        {
            int index = blockIdx.x * blockDim.x + threadIdx.x;
            if (index > N) return;
            output[index] = index == 0 ? 0 : input[index - 1];
        }
        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int* odata, const int* idata) {

            // TODO
            int* dev1, * dev2;
            cudaMalloc((void**)&dev1, n * sizeof(int));
            cudaMalloc((void**)&dev2, n * sizeof(int));
            cudaMemcpy(dev1, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            nvtxRangePushA("Naive scan");
            timer().startGpuTimer();
            int mxd = ilog2ceil(n);
            for (int d = 1; d <= mxd; d++)
            {
                kernNaiveScan << <(n + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE >> > (n, 1 << (d - 1), dev1, dev2);
                std::swap(dev1, dev2);
            }
            kernNaiveShift << <(n + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE >> > (n, dev1, dev2);
            timer().endGpuTimer();
            nvtxRangePop();
            cudaMemcpy(odata, dev2, n * sizeof(int), cudaMemcpyDeviceToHost);
            cudaFree(dev1);
            cudaFree(dev2);

        }
    }
}

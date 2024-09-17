#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"

#define blockSize 256

namespace StreamCompaction {
    namespace Naive {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }
        // TODO: __global__
        __global__ void kernNaiveScan(int n, int* odata, const int* idata, int offset)
        {
            int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
            if (idx >= n) return;

            if (idx >= offset)
            {
                odata[idx] = idata[idx - offset] + idata[idx];
            }
            else
            {
                odata[idx] = idata[idx];
            }
        }


        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {

            // allocate cuda buffers
            int* dev_obufferA;
            int* dev_obufferB;
            cudaMalloc((void**)&dev_obufferA, n * sizeof(int));
            cudaMemcpy(dev_obufferA, idata, sizeof(int) * n, cudaMemcpyHostToDevice);
            cudaMalloc((void**)&dev_obufferB, n * sizeof(int));
            cudaMemset(dev_obufferB, 0, n * sizeof(int));

            nvtxRangePushA("Naive");
            timer().startGpuTimer();

            dim3 fullBlocksPerGrid((n + blockSize - 1) / blockSize);
            int d = ilog2ceil(n);

            for (int i = 1; i <= d; ++i)
            {
                int offset = 1 << (i - 1);
                kernNaiveScan <<< fullBlocksPerGrid, blockSize >>> (n, dev_obufferB, dev_obufferA, offset);
                std::swap(dev_obufferA, dev_obufferB);
            }

            timer().endGpuTimer();
            nvtxRangePop();

            // result is in A
            // do right shift
            odata[0] = 0;
            cudaMemcpy(odata + 1, dev_obufferA, sizeof(int) * (n - 1), cudaMemcpyDeviceToHost);

            cudaFree(dev_obufferA);
            cudaFree(dev_obufferB);
            
        }
    }
}

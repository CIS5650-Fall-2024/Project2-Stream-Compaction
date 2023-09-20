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

        __global__ void KernNaiveScan(int n, int d, int* odata, int* idata)
        {
            int index = threadIdx.x + (blockIdx.x * blockDim.x);
            if (index > n) return;
            int path = 1 << (d - 1);
            if (index >= path)
            {
                odata[index] = idata[index - path] + idata[index];
            }
            else
            {
                odata[index] = idata[index];
            }
        }

        void scan(int n, int* odata, const int* idata) {
            dim3 BlockDim((n + BlockSize - 1) / BlockSize);

            int* dev_odata;
            int* dev_odata2;
            cudaMalloc((void**)&dev_odata2, n * sizeof(int));
            checkCUDAError("cudaMalloc dev_odata1 failed!");
            cudaMalloc((void**)&dev_odata, n * sizeof(int));
            checkCUDAError("cudaMalloc dev_odata2 failed!");

            cudaMemcpy(dev_odata, idata, sizeof(int) * n, cudaMemcpyHostToDevice);
            cudaMemcpy(dev_odata2, idata, sizeof(int) * n, cudaMemcpyHostToDevice);

            int depth = ilog2ceil(n);

            timer().startGpuTimer();

            for (int d = 1; d <= depth; ++d) {
                KernNaiveScan <<<BlockDim, BlockSize >>> (n, d, dev_odata, dev_odata2);
                cudaDeviceSynchronize();
                int* temp = dev_odata2;
                dev_odata2 = dev_odata;
                dev_odata = temp;
            }

            timer().endGpuTimer();
            cudaMemcpy(odata + 1, dev_odata2, (n - 1) *sizeof(int), cudaMemcpyDeviceToHost);
            odata[0] = 0;
            cudaFree(dev_odata);
            cudaFree(dev_odata2);
        }

    }
}

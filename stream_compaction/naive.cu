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
        __global__ void kernNaiveScan(const int n, const int d, int* odata, int* idata)
        {
            int index = blockIdx.x * blockDim.x + threadIdx.x;
            if (index >= n)return;
            if (index >= d)
                odata[index] = idata[index - d] + idata[index];
            else
                odata[index] = idata[index];
        }
        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int* odata, const int* idata) {
            int dMax = ilog2ceil(n);
            int* dev_idata, * dev_odata;
            cudaMalloc((void**)&dev_idata, n * sizeof(int));
            cudaMalloc((void**)&dev_odata, n * sizeof(int));
            cudaMemcpy(dev_idata+1, idata, (n - 1) * sizeof(int), cudaMemcpyHostToDevice);
            dim3 fullBlocksPerGrid((n + blockSize - 1) / blockSize);
            timer().startGpuTimer();
            // DONE
            for (int i = 1; i <= dMax; i++)
            {
                kernNaiveScan << <fullBlocksPerGrid, blockSize >> > (n, 1 << (i - 1), dev_odata, dev_idata);
                std::swap(dev_odata, dev_idata);
            }
            timer().endGpuTimer();
            cudaMemcpy(odata, dev_idata, n*sizeof(int), cudaMemcpyDeviceToHost);
            cudaFree(dev_idata);
            cudaFree(dev_odata);
        }
    }
}

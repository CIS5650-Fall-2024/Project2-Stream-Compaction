#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"

__global__ void kernNaiveScan(int offset, int n, int* odata, const int* idata) {
    int k = threadIdx.x + (blockIdx.x * blockDim.x);
    if (k >= n) {
        return;
    }
    odata[k] = idata[k];
    if (k >= offset) {
        odata[k] += idata[k - offset];
    }
}

__global__ void kernRightShift(int n, int* data) {
    int index = threadIdx.x + (blockIdx.x * blockDim.x);
    if (index >= n - 1) {
        return;
    }
    int readVal = data[index];
    __syncthreads();
    data[index + 1] = readVal;
}

namespace StreamCompaction {
    namespace Naive {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }
        // TODO: __global__

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int* odata, const int* idata) {
            int* dev_in;
            int* dev_out;           
            int noOfIters = ilog2ceil(n);
            dim3 fullBlocksPerGrid((n + BLOCKSIZE - 1) / BLOCKSIZE);

            cudaMalloc((void**)&dev_in, n * sizeof(int));
            checkCUDAError("cudaMalloc dev_in failed!");

            cudaMalloc((void**)&dev_out, n * sizeof(int));
            checkCUDAError("cudaMalloc dev_out failed!");            
            
            cudaMemcpy(dev_in, idata, sizeof(int) * n, cudaMemcpyHostToDevice);
            timer().startGpuTimer();
            for (int d = 1; d <= noOfIters; d++) {
                int offset = 1 << (d - 1);
                kernNaiveScan << <fullBlocksPerGrid, BLOCKSIZE >> > (offset, n, dev_out, dev_in);
                std::swap(dev_in, dev_out);
            }
            std::swap(dev_in, dev_out);
            kernRightShift << <fullBlocksPerGrid, BLOCKSIZE >> > (n, dev_out);
            cudaMemcpy(odata, dev_out, sizeof(int) * n, cudaMemcpyDeviceToHost);
            odata[0] = 0;            
            timer().endGpuTimer();
            cudaFree(dev_in);
            cudaFree(dev_out);
        }
    }
}

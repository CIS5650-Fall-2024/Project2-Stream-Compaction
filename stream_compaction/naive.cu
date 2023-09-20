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
        __global__ void kernNaiveScan(int n,int size, int* odata, int* idata) {
            int index = threadIdx.x + (blockIdx.x * blockDim.x);
            if (index >= n) {
                return;
            }
            if (index >= size) {
                odata[index] = idata[index - size] + idata[index];
            }
            else {
                odata[index] = idata[index];
            }
        }
        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            
            // TODO
            int numOfblock = (n + blockSize - 1) / blockSize;
            dim3 threadsPerBlock(blockSize);

            int *buffer1;
            int *buffer2;
            int zerobuffer=0;
            cudaMalloc((void**)&buffer1, n * sizeof(int));
            cudaMalloc((void**)&buffer2, n * sizeof(int));
            
            cudaMemcpy(&(buffer1[1]), idata, (n-1) * sizeof(int),cudaMemcpyHostToDevice);
            cudaMemcpy(buffer1, &zerobuffer, sizeof(int), cudaMemcpyHostToDevice);
            int iter = ilog2ceil(n);
            const int size = 1 << iter;
            timer().startGpuTimer();
            for (int i = 1; i <size; i=i<<1) {
                kernNaiveScan<<<numOfblock, threadsPerBlock >>> (n,i,buffer2,buffer1);
                int *temp = buffer2;
                buffer2 = buffer1;
                buffer1 = temp;
            }
            timer().endGpuTimer();
            cudaMemcpy(odata, buffer1, n * sizeof(int), cudaMemcpyDeviceToHost);
            cudaFree(buffer1);
            cudaFree(buffer2);
            
        }
    }
}

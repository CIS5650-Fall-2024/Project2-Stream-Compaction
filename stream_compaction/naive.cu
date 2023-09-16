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
        __global__ void kernelNaiveScan(int n, int offset,int *odata, const int* idata) {
            int id = threadIdx.x + blockIdx.x * blockDim.x;
            if (id >= n)return;
            if (id >= offset) {
                odata[id] = idata[id] + idata[id - offset];
            }
            else {
                odata[id] = idata[id];
            }
        }


        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            timer().startGpuTimer();
            // TODO
            //initialize
            int* dev_ping;
            int* dev_pong;
            cudaMalloc((void**)&dev_ping, n * sizeof(int));
            checkCUDAError("cudaMalloc dev_ping failed!");
            cudaMalloc((void**)&dev_pong, n * sizeof(int));
            checkCUDAError("cudaMalloc dev_pong failed!");
            cudaMemcpy(dev_ping, idata, sizeof(int) * n, cudaMemcpyHostToDevice);
            
            int* dev_in;
            int* dev_out;
            int blockSize = 32;
            dim3 fullBlocksPerGrid((n + blockSize - 1) / blockSize);
            int layerCnt = ilog2ceil(n);
            int offset = 1;
            for (int i = 0;i < layerCnt;++i) {
                if (i % 2 == 0) {
                    dev_in = dev_ping;
                    dev_out = dev_pong;
                }else {
                    dev_in = dev_pong;
                    dev_out = dev_ping;
                }
                kernelNaiveScan << <fullBlocksPerGrid, blockSize >> > (n,offset,dev_out, dev_in);
                offset *= 2;
            }

            //exclusive scan
            cudaMemcpy(odata + 1, dev_out, sizeof(int) * (n-1), cudaMemcpyDeviceToHost);
            if(n>0)odata[0] = 0;
            //free mem
            cudaFree(dev_ping);
            cudaFree(dev_pong);

            timer().endGpuTimer();
        }
    }
}

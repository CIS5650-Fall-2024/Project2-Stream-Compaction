#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"

#define block_size 256

namespace StreamCompaction {
    namespace Naive {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }
        // TODO: __global__
        __global__ void scan_global(int n, int *odata, int *idata, int *temp)
        {
            int thid = threadIdx.x + (blockIdx.x * blockDim.x);
            int pout = 0, pin = 1;
            // Load input into global memory.
            // This is exclusive scan, so shift right by one
            // and set first element to 0
            
            temp[pout * n + thid] = (thid > 0) ? idata[thid - 1] : 0;
            __syncthreads();
            for (int offset = 1; offset < n; offset*=2) {
                pout = 1 - pout; // swap double buffer indices
                pin = 1 - pout;
                if (thid >= offset)
                    temp[pout * n + thid] = temp[pin * n + thid - offset] + temp[pin* n + thid];
                else
                    temp[pout * n + thid] = temp[pin * n + thid];
                __syncthreads();
            }
            odata[thid] = temp[pout * n + thid]; // write output
        }
        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            timer().startGpuTimer();
            // TODO
            int *g_odata,*g_idata,*temp;
            cudaError_t result = cudaMalloc((void**)&g_idata, n * sizeof(int));
            result = cudaMalloc((void**)&g_odata,n*sizeof(int));
            result = cudaMalloc((void**)&temp,2 * n * sizeof(int));
            cudaMemcpy(g_idata,idata,sizeof(int)*n,cudaMemcpyHostToDevice);
            int threadsPerBlock = 256;
            int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
            scan_global<<<blocksPerGrid,threadsPerBlock>>>(n,g_odata,g_idata,temp);
            cudaMemcpy(odata,g_odata,sizeof(int)*n,cudaMemcpyDeviceToHost);
            for (int i = 0; i < n; i++)
            {
                printf("%d %d\n", idata[i], odata[i]);
            }
            timer().endGpuTimer();
        }
    }
}

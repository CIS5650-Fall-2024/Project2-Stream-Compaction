#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"
#include "device_launch_parameters.h"

#define BLOCKSIZE 128

namespace StreamCompaction {
    namespace Naive {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }
        /**
        * Parallel naive scan
        */
        __global__ void kernNaiveScan(int n, int i, int* odata, int* idata) {
            int idx = threadIdx.x + (blockIdx.x * blockDim.x);
            if (idx < n) {
                // if idx >= 2^(i-1)
                int k = 1 << i - 1;
                odata[idx] = idx >= k ? (idata[idx - k] + idata[idx]) : idata[idx];
            }
        }
        /**
        * After inclusive scan, make the array exclusive
        */
        __global__ void kernMakeExclusive(int n, int* odata, int* idata) {
            int idx = threadIdx.x + (blockIdx.x * blockDim.x);
            if (idx < n) {
                odata[idx] = idx == 0 ? 0 : idata[idx - 1];
            }
        }
        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            dim3 blockPerGrid((n + BLOCKSIZE - 1) / BLOCKSIZE);
            int* dev_odata;
            int* dev_idata;
            cudaMalloc((void**)&dev_odata, n * sizeof(int));
            checkCUDAErrorFn("cudaMalloc dev_odata failed!");
            cudaMalloc((void**)&dev_idata, n * sizeof(int));
            checkCUDAErrorFn("cudaMalloc dev_idata failed!");

            cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            checkCUDAErrorFn("cudaMemcpy dev_idata failed!");
            cudaDeviceSynchronize();

            timer().startGpuTimer();
            for (int i = 1; i <= ilog2ceil(n); ++i) {
                kernNaiveScan << <blockPerGrid, BLOCKSIZE >> > (n, i, dev_odata, dev_idata);
                std::swap(dev_idata, dev_odata);
            }
            kernMakeExclusive << <blockPerGrid, BLOCKSIZE >> > (n, dev_odata, dev_idata);
            cudaDeviceSynchronize();
            checkCUDAErrorFn("kernMakeExclusive failed!");
            timer().endGpuTimer();
            
            cudaMemcpy(odata, dev_odata, n * sizeof(int), cudaMemcpyDeviceToHost);
            checkCUDAErrorFn("cudaMemcpy dev_idataToodata failed!");

            cudaFree(dev_odata);
            cudaFree(dev_idata);
        }
    }
}

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

        int* dev_idata; // input data
        int* dev_odata; // output data
        int* dev_tdata; // temp data

        // Define the blockSize
        int blockSize = 128;

        // DONE: __global__
        __global__ void naiveParallelScanKernel(int n, int *odata, const int *idata, const int offset)
        {
            // Using the Naive algorithm from GPU Gems 3, Section 39.2.1.
            int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
            
            if(idx >= n)
            {
                return;
            }

            if (idx >= offset)
            {
                odata[idx] = idata[idx - offset] + idata[idx];
            }
            else
            {
                odata[idx] = idata[idx];
            }
        }

        // Insert the identity and shift right
        __global__ void makeExclusive(int n, int* odata, int* idata) {
            int index = threadIdx.x + (blockIdx.x * blockDim.x);
            if (index >= n) {
                return;
            }
            if (index == 0) {
                odata[index] = 0;
            }
            else {
                odata[index] = idata[index - 1];
            }
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            // DONE

            // Memory allocation for input data
            cudaMalloc((void**)&dev_idata, n * sizeof(int));
            checkCUDAErrorFn("cudaMalloc dev_idata failed!");
            
            // Memory allocation for output data
            cudaMalloc((void**)&dev_odata, n * sizeof(int));
            checkCUDAErrorFn("cudaMalloc dev_odata failed!");

            // Copy from host input data to device input data
            cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            checkCUDAErrorFn("cudaMemcpy to device failed!");

            dim3 fullBlocksPerGrid((n + blockSize - 1) / blockSize);

            // ilog2ceil(x): computes the ceiling of log2(x) as an integer
            int numLevels = ilog2ceil(n);
            int offset;

            timer().startGpuTimer();

            for (int d = 1; d <= numLevels; ++d)
            {
                offset = 1 << (d - 1);
                naiveParallelScanKernel<<<fullBlocksPerGrid, blockSize>>> (n, offset, dev_odata, dev_idata);
                std::swap(dev_idata, dev_odata);
            }

            // Insert the identity and shift to the right to make exclusive
            makeExclusive<<<fullBlocksPerGrid, blockSize>>> (n, dev_odata, dev_idata);

            timer().endGpuTimer();

            cudaMemcpy(odata, dev_odata, n * sizeof(int), cudaMemcpyDeviceToHost);
            checkCUDAErrorFn("cudaMempcy to host failed!");
            
            // Free the dev data
            cudaFree(dev_idata);
            cudaFree(dev_odata);
            checkCUDAErrorFn("cudaFree failed!");
        }
    }
}

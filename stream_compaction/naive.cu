#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"
/*! Block size used for CUDA kernel launch. */
#define blockSize 128

namespace StreamCompaction {
    
    namespace Naive {
        using StreamCompaction::Common::PerformanceTimer;

        int* dev_idata;
        int* dev_odata;

        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        void initScan(int N, int* odata, const int* idata)
        {
            int shift = 0;
            cudaMalloc((void**)&dev_idata, N * sizeof(int));
            cudaMemcpy(dev_idata, &(shift), sizeof(int), cudaMemcpyHostToDevice);
            cudaMemcpy(&dev_idata[1], idata, sizeof(int) * (N - 1), cudaMemcpyHostToDevice);
            checkCUDAError("cudaMalloc dev_idata failed!", __LINE__);
            cudaMalloc((void**)&dev_odata, N * sizeof(int));
            cudaMemcpy(dev_odata, &(shift), sizeof(int), cudaMemcpyHostToDevice);
            cudaMemcpy(&dev_odata[1], idata, sizeof(int) * (N - 1), cudaMemcpyHostToDevice);
            checkCUDAError("cudaMalloc dev_odata failed!", __LINE__);
            cudaDeviceSynchronize();
        }

        void endScan()
        {
            cudaFree(dev_idata);
            cudaFree(dev_odata);
        }

        __global__ void NaiveMapKernel(int n, int* idata, int* odata)
        {
            int index = (blockIdx.x * blockDim.x) + threadIdx.x;
            if (index >= n)
            {
                return;
            }
            if (odata[index] != 0)
            {
                idata[index] = 1;
            }
            else
            {
                idata[index] = 0;
            }
        }

        __global__ void NaiveScanKernel(int n, int interval, int* odata, int* idata)
        {
            int index = (blockIdx.x * blockDim.x) + threadIdx.x;
            if (index >= n)
            {
                return;
            }
            
            if (index >= interval)
            {
                odata[index] = idata[index] + idata[index - interval];
            }
            else
            {
                odata[index] = idata[index];
            }
        }
        
        // TODO: __global__

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
    
        void scan(int n, int *odata, const int *idata) {
            
            // TODO
            initScan(n, odata, idata);
            cudaDeviceSynchronize();
            timer().startGpuTimer();
            dim3 fullBlocksPerGrid((n + blockSize - 1) / blockSize);
            int blocks = 1;
            int round = ilog2ceil(n);

            cudaDeviceSynchronize();
            for (int i = 0; i < round; i++)
            {
                int interval = int(powf(2, i)); 
                NaiveScanKernel << <fullBlocksPerGrid, blockSize >> > (n, interval, dev_odata, dev_idata);
                cudaDeviceSynchronize();
                std::swap(dev_idata, dev_odata);
            }
            std::swap(dev_idata, dev_odata);
            timer().endGpuTimer();
            cudaMemcpy(odata, dev_odata, sizeof(int) * n, cudaMemcpyDeviceToHost);
            endScan();
        }
    }
   
}

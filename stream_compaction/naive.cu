#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"
#include <iostream>

#define blocksize 512

namespace StreamCompaction {
    namespace Naive {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }
        __global__ void kernNaiveScanStep(int n, int d, int *odata, int *idata){
            int k = threadIdx.x + blockIdx.x * blockDim.x;
            int backward_ind = k - d;
            if (k >= n) {
                return;
            }

            if(backward_ind < 0){
                odata[k] = idata[k]; 
            } else {
                odata[k] = idata[k] + idata[backward_ind];
            }

        }

        void printArray(int arr[], int size) {
            for(int i = 0; i < size; i++) {
                std::cout << arr[i] << " ";
            }
            std::cout << std::endl;
        }
        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */

        void scan(int n, int *odata, const int *idata) {

            int *dev_idata, *dev_odata;
            cudaMalloc((void**)&dev_idata, n * sizeof(int));
            cudaMalloc((void**)&dev_odata, n * sizeof(int));
            cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);

            timer().startGpuTimer();

            int numBlocks = (n + (blocksize - 1)) / blocksize;
            int d = 1;
            for(int i = 0; i < ilog2ceil(n); ++i){
                kernNaiveScanStep<<<numBlocks, blocksize>>>(n, d, dev_odata, dev_idata);
                cudaDeviceSynchronize();
                /*
                cudaError_t err = cudaGetLastError();
                if (err != cudaSuccess) {
                    std::cout << "CUDA Error: " << cudaGetErrorString(err) << std::endl;
                }
                */

                d *= 2;

                int *tmp = dev_idata;
                dev_idata = dev_odata;
                dev_odata = tmp;

                /*
                odata[0] = 0;
                cudaMemcpy(odata + 1, dev_idata, (n-1) * sizeof(int), cudaMemcpyDeviceToHost);

                printArray(odata, n);
                */
            }


            timer().endGpuTimer();
            odata[0] = 0;
            cudaMemcpy(odata + 1, dev_idata, (n-1) * sizeof(int), cudaMemcpyDeviceToHost);
        }
    }
}

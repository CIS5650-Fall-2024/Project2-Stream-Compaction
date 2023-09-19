#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"



#define blockSize 128
namespace StreamCompaction {
    namespace Naive {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }
        // TODO: __global__
        __global__ void scanOperation(int n, int base, int* odata, int* idata) {
            int idx = (blockIdx.x * blockDim.x) + threadIdx.x;

            if(idx >= n){
                return;

            }
             
            int curr_pos = 1 << (base-1);

            if (idx < curr_pos) {
            
                odata[idx] = idata[idx];
            }
            else {
                odata[idx] = idata[idx - curr_pos] + idata[idx];
            
            }

        
        
        
        }
        
        __global__ void convertToExclusive(int n, int* odata, int* idata) {
            int idx = (blockIdx.x * blockDim.x) + threadIdx.x;

            if (idx >= n) {
                return;

            }
            if (idx == 0) {
                
                odata[idx] = 0;
                return;
            }
            
            odata[idx] = idata[idx - 1];


        
        }






        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            timer().startGpuTimer();
            // TODO
            int* gpu_odata;
            int* gpu_idata;

            cudaMalloc((void**)&gpu_odata, n * sizeof(int));
            cudaMalloc((void**)&gpu_idata, n * sizeof(int));
            cudaMemcpy(gpu_idata, idata, sizeof(int) * n, cudaMemcpyHostToDevice);

            checkCUDAError("memory error!!!!!");

            dim3 fullBlocksPerGrid((n + blockSize - 1) / blockSize);
            dim3 numBlocks((n -1 + blockSize - 1) / blockSize);
            for (int i = 1; i <= ilog2ceil(n); i++) {
                scanOperation <<<fullBlocksPerGrid, blockSize >>> (n, i, gpu_odata, gpu_idata);

                checkCUDAError("error in loop!!!!!");
                int* temp = gpu_odata;
                gpu_odata = gpu_idata;
                gpu_idata = temp;


            }

            
            convertToExclusive << <numBlocks, blockSize >> > (n, gpu_odata, gpu_idata);

            cudaMemcpy(odata, gpu_odata, sizeof(int) * n, cudaMemcpyDeviceToHost);

            checkCUDAError("memory error!!!!!");

            cudaFree(gpu_odata);
            cudaFree(gpu_idata);


            timer().endGpuTimer();
        }
    }
}

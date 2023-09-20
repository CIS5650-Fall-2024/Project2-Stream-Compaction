#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"

// 48kb shmem per threadblock. What should block size be?

namespace StreamCompaction {
    namespace Naive {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

#if GLOBAL_MEM
        int* dev_buf1;
        int* dev_buf2;

        // TODO: __global__
        __global__ void kernArrayShiftRight(int N, int* odata, const int* idata) {
            int index = threadIdx.x + (blockIdx.x * blockDim.x);
            if (index >= N) {
                return;
            }
            odata[index] = (index == 0) ? 0 : idata[index - 1];
        }

        __global__ void kernArrayAddWithOffset(int N, int offset, int* odata, const int* idata) {
            int index = threadIdx.x + (blockIdx.x * blockDim.x);
            if (index >= N) {
                return;
            }
            odata[index] = idata[index];
            if (index >= offset) {
                odata[index] += idata[index - offset];
            }
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int* odata, const int* idata) {
            int N = 1 << ilog2ceil(n);
            cudaMalloc((void**)&dev_buf1, N * sizeof(int));
            checkCUDAErrorFn("malloc dev_buf1 failed");
            cudaMalloc((void**)&dev_buf2, N * sizeof(int));
            checkCUDAErrorFn("malloc dev_buf2 failed");
            cudaMemcpy(dev_buf1, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            checkCUDAErrorFn("copy idata to dev_buf1 failed");

            dim3 fullBlocksPerGrid((N + BLOCK_SIZE - 1) / BLOCK_SIZE);

            timer().startGpuTimer();
            // TODO
            for (int d = 1; d < N; d <<= 1) {
                kernArrayAddWithOffset << <fullBlocksPerGrid, BLOCK_SIZE >> > (N, d, dev_buf2, dev_buf1);
                std::swap(dev_buf1, dev_buf2);
            }
            kernArrayShiftRight<<<fullBlocksPerGrid, BLOCK_SIZE>>>(N, dev_buf2, dev_buf1);
            timer().endGpuTimer();

            cudaMemcpy(odata, dev_buf2, n * sizeof(int), cudaMemcpyDeviceToHost);
            checkCUDAErrorFn("copy dev_buf2 to odata failed");
            cudaFree(dev_buf1);
            checkCUDAErrorFn("free dev_buf1 failed");
            cudaFree(dev_buf2);
            checkCUDAErrorFn("free dev_buf2 failed");
        }
#endif
#if SHARED_MEM
        __global__ void scan(float* g_odata, float* g_idata, int n) {
            extern __shared__ float temp[]; // allocated on invocation    
            int index = threadIdx.x + (blockIdx.x * blockDim.x);   
            int pout = 0, pin = 1;   // Load input into shared memory.    
                                     // This is exclusive scan, so shift right by one    
                                     // and set first element to 0   
            temp[pout*n + index] = (index > 0) ? g_idata[index-1] : 0;   
            __syncthreads();   
            for (int offset = 1; offset < n; offset <<= 1)   {     
                pout = 1 - pout; // swap double buffer indices     
                pin = 1 - pout;     
                if (index >= offset)       
                    temp[pout*n+index] += temp[pin*n+index - offset];     
                else       
                    temp[pout*n+index] = temp[pin*n+index];
                __syncthreads();   
            }   
            g_odata[index] = temp[pout*n+index]; // write output 
        } 

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int* odata, const int* idata) {
            timer().startGpuTimer();
            // TODO
            timer().endGpuTimer();
        }
#endif
    }
}

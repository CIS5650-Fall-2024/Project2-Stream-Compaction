#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "radixsort.h"
#include "efficient.h"

namespace StreamCompaction {
	namespace RadixSort {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        const int blockSize = 128;

        __global__ void kernComputeBits(int n, int bit, const int* idata, int* odata) {
            int index = threadIdx.x + blockIdx.x * blockDim.x;
            if (index >= n) return;

            int mask = 1 << bit;
            odata[index] = (idata[index] & mask) ? 1 : 0;
        }

        

        __global__ void kernComplementBits(int n, int total, int* idata, int* odata) {
            int index = threadIdx.x + blockIdx.x * blockDim.x;
            if (index >= n) return;

            odata[index] = index - idata[index] + total;
        }

        __global__ void kernScatter(int n, int* b_array, int* t_array, int* f_array, int* odata) {
            int index = threadIdx.x + blockIdx.x * blockDim.x;
            if (index >= n) return;

            odata[index] = (b_array[index]) ? t_array[index] : f_array[index];
        }


        void sort(int n, int* odata, const int* idata) {

            int* dev_idata;
            int* dev_odata;
            int* dev_b;
            int* dev_e;
            int* dev_f;
            int* dev_t;

            cudaMalloc((void**)&dev_idata, n * sizeof(int));
            cudaMalloc((void**)&dev_odata, n * sizeof(int));
            cudaMalloc((void**)&dev_b, n * sizeof(int));
            cudaMalloc((void**)&dev_e, n * sizeof(int));
            cudaMalloc((void**)&dev_f, n * sizeof(int));
            cudaMalloc((void**)&dev_t, n * sizeof(int));

            cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);


        }




	}
}
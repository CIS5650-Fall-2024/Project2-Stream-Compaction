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

        __global__ void kernComputeBits(int n, int bit, const int* idata, int* odata1, int* odata2) {
            int index = threadIdx.x + blockIdx.x * blockDim.x;
            if (index >= n) return;

            int mask = 1 << bit;
            odata1[index] = (idata[index] & mask) ? 1 : 0;
            odata2[index] = ~odata1[index];
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
            int* dev_d;

            cudaMalloc((void**)&dev_idata, n * sizeof(int));
            cudaMalloc((void**)&dev_odata, n * sizeof(int));
            cudaMalloc((void**)&dev_b, n * sizeof(int));
            cudaMalloc((void**)&dev_e, n * sizeof(int));
            cudaMalloc((void**)&dev_f, n * sizeof(int));
            cudaMalloc((void**)&dev_t, n * sizeof(int));
            cudaMalloc((void**)&dev_d, n * sizeof(int));

            cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);

            
            for (int i = 0; i < sizeof(int); i++) {
                dim3 fullBlocksPerGrid((n +blockSize - 1) / blockSize);

                kernComputeBits<<<fullBlocksPerGrid, blockSize>>>(n, i, dev_idata, dev_b, dev_e);

                StreamCompaction::Efficient::scan(n, dev_e, dev_f);

                int total1 = 0;
                cudaMemcpy(&total1, &dev_e[n - 1], sizeof(int), cudaMemcpyDeviceToHost);
                int total2 = 0;
                cudaMemcpy(&total2, &dev_f[n - 1], sizeof(int), cudaMemcpyDeviceToHost);

                kernComplementBits<<<fullBlocksPerGrid, blockSize>>>(n, total1 + total2, dev_f, dev_t);


                kernScatter<<<fullBlocksPerGrid, blockSize>>>(int n, dev_b, dev_t, dev_f, dev_d);

                StreamCompaction::Common::kernScatter<<<fullBlocksPerGrid, blockSize>>>(n, dev_odata, dev_idata, dev_b, dev_d);


                int* temp = dev_idata;
                dev_idata = dev_odata;
                dev_odata = temp;
            }
            cudaMemcpy(odata, dev_idata, n * sizeof(int), cudaMemcpyDeviceToHost);

            cudaFree(dev_idata);
            cudaFree(dev_odata);
            cudaFree(dev_b);
            cudaFree(dev_e);
            cudaFree(dev_f);
            cudaFree(dev_t);
            cudaFree(dev_d);


        }




	}
}
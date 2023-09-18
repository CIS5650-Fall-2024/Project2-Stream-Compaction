#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "radix_sort.h"
#include "efficient_sharedmem.h"
#define BLOCK_SIZE 1024
namespace StreamCompaction {
    namespace RadixSort {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        __global__ void kernCheckZeroBit(int N, int bitmask, const int* input, int* output)
        {
            int index = blockIdx.x * blockDim.x + threadIdx.x;
            if (index >= N) return;
            output[index] = !(input[index] & bitmask);
        }

        __global__ void kernGenerateIndices(int N, int totalFalses, int* e, int* f, int* outIndices)
        {
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i >= N) return;
            outIndices[i] = e[i] ? f[i] : i - f[i] + totalFalses;
        }
        
        void sort(int n, int* odata, const int* idata) {
            int* dev_i1, * dev_e, * dev_i2, * dev_i3;
            cudaMalloc((void**)&dev_i1, n * sizeof(int));
            cudaMalloc((void**)&dev_i2, n * sizeof(int));
            cudaMalloc((void**)&dev_i3, n * sizeof(int));
            cudaMalloc((void**)&dev_e, n * sizeof(int));
            cudaMemcpy(dev_i1, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            EfficientSharedMem::gpuScanTempBuffer tmpBuf(n, BLOCK_SIZE, nullptr);
            int*& dev_f = tmpBuf.buffers[0].first;
            timer().startGpuTimer();
            for (int i=0;i<31;i++)
            {
                int mask = (1 << i);
                int nblocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
                kernCheckZeroBit << < nblocks, BLOCK_SIZE >> > (n, mask, dev_i1, dev_e);
                checkCUDAError("kernCheckZeroBit error");
                cudaMemcpy(dev_f, dev_e, sizeof(int) * n, cudaMemcpyDeviceToDevice);
                EfficientSharedMem::gpuScanWorkEfficientOptimized(tmpBuf);
                int totalFalses,lastE;
                cudaMemcpy(&totalFalses, dev_f + n - 1, sizeof(int), cudaMemcpyDeviceToHost);
                cudaMemcpy(&lastE, dev_e + n - 1, sizeof(int), cudaMemcpyDeviceToHost);
                totalFalses += lastE;
                kernGenerateIndices << < nblocks, BLOCK_SIZE >> > (n, totalFalses, dev_e, dev_f, dev_i2);
                checkCUDAError("kernGenerateIndices error");
                Common::kernScatter << < nblocks, BLOCK_SIZE >> > (n, dev_i2, dev_i1, dev_i3, false);
                checkCUDAError("kernScatter error");
                std::swap(dev_i1, dev_i3);
            }
            timer().endGpuTimer();
            cudaMemcpy(odata, dev_i1, sizeof(int) * n, cudaMemcpyDeviceToHost);
            cudaFree(dev_i1);
            cudaFree(dev_i2);
            cudaFree(dev_i3);
            cudaFree(dev_e);
        }

    }
}

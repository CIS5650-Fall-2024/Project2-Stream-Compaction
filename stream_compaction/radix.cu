#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"
#include "radix.h"

#include <device_launch_parameters.h>
#include <thrust/device_ptr.h>

#define getBit(num, k) ((num) & (1 << (k))) >> (k)

constexpr int blockSize = 256;

namespace StreamCompaction
{
    namespace Radix
    {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        __global__ void kernGetBitMask(int N, int k, const int* data, int* b, int* e)
        {
            int index = (blockIdx.x * blockDim.x) + threadIdx.x;
            if (index >= N) { return; }

            int bit = getBit(data[index], k);
            b[index] = bit;
            e[index] = bit ^ 1;
        }

        __global__ void kernGetIdx(int N, int totalFalses, const int* b, const int* f, int* d)
        {
            int index = (blockIdx.x * blockDim.x) + threadIdx.x;
            if (index >= N) { return; }

            d[index] = b[index] ? index - f[index] + totalFalses : f[index];
        }

        void sort(int n, int* odata, const int* idata)
        {
            sort(n, 31, odata, idata);
        }

        void sort(int n, int bits, int* odata, const int* idata)
        {
            int nCeil = 1 << ilog2ceil(n);

            int* dev_idata;
            cudaMalloc((void**)&dev_idata, n * sizeof(int));
            checkCUDAError("cudaMalloc dev_data failed!");

            int* dev_odata;
            cudaMalloc((void**)&dev_odata, n * sizeof(int));
            checkCUDAError("cudaMalloc dev_data failed!");

            int* dev_b;
            cudaMalloc((void**)&dev_b, n * sizeof(int));
            checkCUDAError("cudaMalloc dev_b failed!");

            int* dev_e;
            cudaMalloc((void**)&dev_e, n * sizeof(int));
            checkCUDAError("cudaMalloc dev_e failed!");

            int* dev_f;
            cudaMalloc((void**)&dev_f, nCeil * sizeof(int));
            checkCUDAError("cudaMalloc dev_f failed!");

            int* dev_d;
            cudaMalloc((void**)&dev_d, n * sizeof(int));
            checkCUDAError("cudaMalloc dev_d failed!");

            thrust::device_ptr<int> dev_thrust_f(dev_f);
            thrust::device_ptr<int> dev_thrust_e(dev_e);

            cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            checkCUDAError("cudaMemcpy idata->dev_idata failed!");

            timer().startGpuTimer();
            for (int k = 0; k < bits; ++k)
            {
                // k-th pass
                dim3 threadsPerBlock(blockSize);
                dim3 blocksPerGrid((n + blockSize - 1) / blockSize);
                kernGetBitMask <<<blocksPerGrid, threadsPerBlock>>> (n, k, dev_idata, dev_b, dev_e);

                cudaMemcpy(dev_f, dev_e, n * sizeof(int), cudaMemcpyDeviceToDevice);
                checkCUDAError("cudaMemcpy dev_e->dev_f failed!");
                Efficient::scan_gpu(nCeil, dev_f);

                int totalFalses = dev_thrust_e[n - 1] + dev_thrust_f[n - 1];

                kernGetIdx <<<blocksPerGrid, threadsPerBlock>>> (n, totalFalses, dev_b, dev_f, dev_d);

                Common::kernScatter <<<blocksPerGrid, threadsPerBlock>>> (n, dev_odata, dev_idata, dev_d);

                std::swap(dev_odata, dev_idata);
            }
            timer().endGpuTimer();

            cudaMemcpy(odata, dev_idata, n * sizeof(int), cudaMemcpyDeviceToHost);
            checkCUDAError("cudaMemcpy dev_idata->odata failed!");

            cudaFree(dev_idata);
            cudaFree(dev_odata);
            cudaFree(dev_b);
            cudaFree(dev_e);
            cudaFree(dev_f);
            cudaFree(dev_d);
        }
    }
}
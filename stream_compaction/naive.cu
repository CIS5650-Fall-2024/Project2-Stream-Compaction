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

        __global__ void kernalNaiveScan(int n, int start, int* write_arr, const int* read_arr)
        {
            int index = threadIdx.x + (blockDim.x * blockIdx.x);
            if (index >= n) return;
            int id = index + start;
            write_arr[id] = read_arr[index] + read_arr[id];
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            // Malloc necessary space on GPU
            int* dev_read;
            int* dev_write;

            cudaMalloc((void**)&dev_read, n * sizeof(int));
            checkCUDAError("cudaMalloc dev_read failed!");
            cudaMalloc((void**)&dev_write, n * sizeof(int));
            checkCUDAError("cudaMalloc dev_write failed!");

            // copy data from CPU to GPU
            cudaMemcpy(dev_read, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            checkCUDAError("Memcpy idata(host) to dev_read(device) failed!");

            // naive scan
            timer().startGpuTimer();
            int d = 1;
            while (d < n)
            {
                int k = n - d;
                kernalNaiveScan<<<(k + BlockSize - 1) / BlockSize, BlockSize >> >(k, d, dev_write, dev_read);
                checkCUDAError("Luanch kernalNaiveScan failed!");
                cudaMemcpy(dev_read + d, dev_write + d, k * sizeof(int), cudaMemcpyDeviceToDevice);

                d <<= 1;
            }
            timer().endGpuTimer();

            // copy data back to CPU
            odata[0] = 0;
            cudaMemcpy(odata + 1, dev_read, (n - 1) * sizeof(int), cudaMemcpyDeviceToHost);
            checkCUDAError("Memcpy dev_write(device) to odata(host) failed!");

            // free memory on GPU
            cudaFree(dev_read);
            cudaFree(dev_write);
        }
    }
}

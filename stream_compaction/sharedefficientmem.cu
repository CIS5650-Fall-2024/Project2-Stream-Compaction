#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include "common.h"
#include "sharedefficientmem.h"



namespace StreamCompaction {
    namespace SharedEfficientMem {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        __global__ void scanSharedEfficientKernel(int* odata, const int* idata, int n) {
            extern __shared__ int temp[]; // allocated on invocation
            int thid = threadIdx.x;
            int offset = 1;

            int ai = thid;
            int bi = thid + (n / 2);

            // Load input into shared memory
            temp[ai] = idata[ai];
            temp[bi] = idata[bi];

            // Up-Sweep (Reduce) phase
            for (int d = n >> 1; d > 0; d >>= 1) {
                __syncthreads();
                if (thid < d) {
                    int ai = offset * (2 * thid + 1) - 1;
                    int bi = offset * (2 * thid + 2) - 1;
                    temp[bi] += temp[ai];
                }
                offset *= 2;
            }

            // Clear the last element
            if (thid == 0) {
                temp[n - 1] = 0;
            }

            // Down-Sweep phase
            for (int d = 1; d < n; d *= 2) {
                offset >>= 1;
                __syncthreads();
                if (thid < d) {
                    int ai = offset * (2 * thid + 1) - 1;
                    int bi = offset * (2 * thid + 2) - 1;

                    int t = temp[ai];
                    temp[ai] = temp[bi];
                    temp[bi] += t;
                }
            }
            __syncthreads();

            // Write results to device memory
            odata[ai] = temp[ai];
            odata[bi] = temp[bi];
        }

        void scanSharedEfficient(int n, int* odata, const int* idata) {
            if (n > 1024 || (n & (n - 1)) != 0) {
                std::cerr << "[ERROR] scanSharedEfficient only supports up to 1024 elements and power-of-two sizes." << std::endl;
                return;
            }

            int* dev_idata = nullptr;
            int* dev_odata = nullptr;

            cudaMalloc(&dev_idata, n * sizeof(int));
            cudaMalloc(&dev_odata, n * sizeof(int));
            cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);

            int blockSize = n / 2;
            int sharedMemSize = n * sizeof(int);

            timer().startGpuTimer();
            scanSharedEfficientKernel<<<1, blockSize, sharedMemSize>>>(dev_odata, dev_idata, n);
            timer().endGpuTimer();

            cudaMemcpy(odata, dev_odata, n * sizeof(int), cudaMemcpyDeviceToHost);

            cudaFree(dev_idata);
            cudaFree(dev_odata);
        }



        __global__ void kernMapToBoolean(int n, int* bools, const int* idata) {
            int i = threadIdx.x + blockIdx.x * blockDim.x;
            if (i >= n) return;
            bools[i] = (idata[i] != 0) ? 1 : 0;
        }

        __global__ void kernScatter(int n, int* odata, const int* idata, const int* bools, const int* indices) {
            int i = threadIdx.x + blockIdx.x * blockDim.x;
            if (i >= n) return;
            if (bools[i] == 1) {
                odata[indices[i]] = idata[i];
            }
        }

        int compactEfficient(int n, int* odata, const int* idata) {
            if (n > 1024 || (n & (n - 1)) != 0) {
                std::cerr << "[ERROR] compactEfficient only supports up to 1024 power-of-two elements." << std::endl;
                return 0;
            }

            int* dev_idata, * dev_bools, * dev_indices, * dev_odata;
            cudaMalloc(&dev_idata, n * sizeof(int));
            cudaMalloc(&dev_bools, n * sizeof(int));
            cudaMalloc(&dev_indices, n * sizeof(int));
            cudaMalloc(&dev_odata, n * sizeof(int));

            cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);

            int blockSize = 1024;
            int numBlocks = (n + blockSize - 1) / blockSize;

            timer().startGpuTimer();

            // 1. Map to boolean
            kernMapToBoolean<<<numBlocks, blockSize>>>(n, dev_bools, dev_idata);
            cudaDeviceSynchronize();

            // 2. Scan boolean array
            scanSharedEfficient(n, dev_indices, dev_bools);
            cudaDeviceSynchronize();

            // 3. Scatter
            kernScatter<<<numBlocks, blockSize>>>(n, dev_odata, dev_idata, dev_bools, dev_indices);
            cudaDeviceSynchronize();

            timer().endGpuTimer();

            int lastBool, lastIndex;
            cudaMemcpy(&lastBool, dev_bools + n - 1, sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(&lastIndex, dev_indices + n - 1, sizeof(int), cudaMemcpyDeviceToHost);
            int total = lastBool + lastIndex;

            cudaMemcpy(odata, dev_odata, total * sizeof(int), cudaMemcpyDeviceToHost);

            cudaFree(dev_idata);
            cudaFree(dev_bools);
            cudaFree(dev_indices);
            cudaFree(dev_odata);

            return total;
        }
    }
}

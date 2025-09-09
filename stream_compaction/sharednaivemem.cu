#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include "common.h"
#include "sharednaivemem.h"

namespace StreamCompaction {
    namespace SharedNaiveMem {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        __global__ void scanSharedNaiveKernel(int* odata, const int* idata, int n) {
            extern __shared__ int temp[];  // size = 2 * blockDim.x * sizeof(int)
            int tid = threadIdx.x;
            int pout = 0, pin = 1;

            if (tid < n) {
                // exclusive scan: shift input right and insert 0 at beginning
                temp[pout * n + tid] = (tid > 0) ? idata[tid - 1] : 0;
            }
            __syncthreads();

            for (int offset = 1; offset < n; offset *= 2) {
                pout = 1 - pout;  // swap ping-pong buffers
                pin = 1 - pout;

                if (tid < n) {
                    if (tid >= offset)
                        temp[pout * n + tid] = temp[pin * n + tid - offset] + temp[pin * n + tid];
                    else
                        temp[pout * n + tid] = temp[pin * n + tid];
                }
                __syncthreads();
            }

            if (tid < n) {
                odata[tid] = temp[pout * n + tid];
            }
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


        void scanSharedNaive(int n, int* odata, const int* idata) {
            if (n > 1024) {
                std::cerr << "[ERROR] scanSharedNaive only supports up to 1024 elements (1 block)." << std::endl;
                return;
            }

            int* dev_idata = nullptr;
            int* dev_odata = nullptr;

            cudaMalloc(&dev_idata, n * sizeof(int));
            cudaMalloc(&dev_odata, n * sizeof(int));
            cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);

            int blockSize = n;
            int sharedMemSize = 2 * blockSize * sizeof(int);  // double buffer

            timer().startGpuTimer();
            scanSharedNaiveKernel<<<1, blockSize, sharedMemSize>>>(dev_odata, dev_idata, n);
            timer().endGpuTimer();
            cudaMemcpy(odata, dev_odata, n * sizeof(int), cudaMemcpyDeviceToHost);

            cudaFree(dev_idata);
            cudaFree(dev_odata);
        }

        int compactNaive(int n, int* odata, const int* idata) {
            if (n > 1024) {
                std::cerr << "[ERROR] compactNaive only supports up to 1024 elements (1 block)." << std::endl;
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
            scanSharedNaive(n, dev_indices, dev_bools);
            cudaDeviceSynchronize();

            // 3. Scatter
            kernScatter<<<numBlocks, blockSize>>>(n, dev_odata, dev_idata, dev_bools, dev_indices);
            cudaDeviceSynchronize();

            timer().endGpuTimer();

            // Copy back
            int lastBool = 0, lastIndex = 0;
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


    } // namespace SharedNaiveMem
} // namespace StreamCompaction

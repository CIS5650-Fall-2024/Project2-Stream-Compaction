#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "radix-sort.h"

#define blockSize 128

namespace StreamCompaction {
    namespace RadixSort {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        __global__ void bitonicMerge(int* data, int n, int j, int k) {
            int i = threadIdx.x + (blockIdx.x * blockDim.x);
            int ixj = i ^ j;

            if (ixj > i) {
                if ((i & k) == 0) {
                    if (data[i] > data[ixj]) {
                        int temp = data[i];
                        data[i] = data[ixj];
                        data[ixj] = temp;
                    }
                } else {
                    if (data[i] < data[ixj]) {
                        int temp = data[i];
                        data[i] = data[ixj];
                        data[ixj] = temp;
                    }
                }
            }
        }

        __global__ void runSortChunk(int n, int numBits, int *odata, const int *idata) {
            extern __shared__ int shared_data[];
            int* shared_bools = (int*)&shared_data[blockDim.x];
            int* shared_scan = (int*)&shared_bools[blockDim.x];

            int index = threadIdx.x;
            int chunkSize = blockDim.x;
            int globalIndex = threadIdx.x + (blockIdx.x * blockDim.x);
            if (globalIndex >= n) {
                return;
            }

            shared_data[index] = idata[globalIndex];
            __syncthreads();

            for (int bit = 0; bit < numBits; bit++) {
                shared_bools[index] = (shared_data[index] & (1 << bit)) ? 0 : 1;
                shared_scan[index] = shared_bools[index];
                __syncthreads();

                // up sweep
                int val = (index + 1) << 1;
                for (int d = 1; d < chunkSize; d <<= 1) {
                    int elemIndex = val * d - 1;
                    if (elemIndex < chunkSize) {
                        shared_scan[elemIndex] += shared_scan[elemIndex - d];
                    }
                    __syncthreads();
                }

                if (index == chunkSize - 1) {
                    shared_scan[chunkSize - 1] = 0;
                }
                __syncthreads();

                // down sweep
                for (int d = chunkSize >> 1; d > 0; d >>= 1) {
                    int elemIndex = val * d - 1;
                    if (elemIndex < chunkSize) {
                        int t = shared_scan[elemIndex - d];
                        shared_scan[elemIndex - d] = shared_scan[elemIndex];
                        shared_scan[elemIndex] += t;
                    }
                    __syncthreads();
                }

                int totalFalses = shared_scan[chunkSize - 1] + shared_bools[chunkSize - 1];

                int targetIndex;
                if (shared_bools[index] == 1) {
                    targetIndex = shared_scan[index];
                } else {
                    targetIndex = index - shared_scan[index] + totalFalses;
                }
                __syncthreads();

                shared_bools[targetIndex] = shared_data[index];
                __syncthreads();

                 shared_data[index] = shared_bools[index];
                __syncthreads();
            }

            odata[globalIndex] = shared_data[index];
        }

        void sort(int n, int *odata, const int *idata) {
            sort(n, -1, odata, idata);
        }

        void sort(int n, int maxVal, int *odata, const int *idata) {
            int nPowOf2 = 1 << ilog2ceil(n);
            int intBits = sizeof(int) * 8;
            int numBits = maxVal < 0 ? intBits : std::min(intBits, ilog2ceil(maxVal));

            int *dev_idata, *dev_odata;
            cudaMalloc((void**)&dev_idata, nPowOf2 * sizeof(int));
            cudaMalloc((void**)&dev_odata, nPowOf2 * sizeof(int));
            checkCUDAError("cudaMalloc failed");
            cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            checkCUDAError("cudaMemcpy failed");
            int numBlocks = (nPowOf2 + blockSize - 1) / blockSize;

            timer().startGpuTimer();
            runSortChunk<<<numBlocks, blockSize, 3 * blockSize * sizeof(int)>>>(nPowOf2, numBits, dev_odata, dev_idata);
            cudaDeviceSynchronize();
    
            for (int k = 2; k <= nPowOf2; k <<= 1) {
                for (int j = k >> 1; j > 0; j >>= 1) {
                    bitonicMerge<<<numBlocks, blockSize>>>(dev_odata, nPowOf2, j, k);
                    cudaDeviceSynchronize();
                }
            }
            timer().endGpuTimer();

            cudaMemcpy(odata, (int*)&dev_odata[nPowOf2 - n], n * sizeof(int), cudaMemcpyDeviceToHost);
            cudaFree(dev_idata);
            cudaFree(dev_odata);
            checkCUDAError("cudaFree failed");
        }
    }
}

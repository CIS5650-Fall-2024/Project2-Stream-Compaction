#include <cuda.h>
#include <cuda_runtime.h>
#include "radix.h"
#include "common.h"
#include "efficient.h" // For scan()
#include <iostream>

namespace StreamCompaction {
    namespace Radix {

        __global__ void kernExtractBit(int n, int bit, int* bitArray, const int* idata) {
            int i = threadIdx.x + blockIdx.x * blockDim.x;
            if (i < n) {
                bitArray[i] = (idata[i] >> bit) & 1;
            }
        }

        __global__ void kernInvert(int n, int* out, const int* in) {
            int i = threadIdx.x + blockIdx.x * blockDim.x;
            if (i < n) {
                out[i] = 1 - in[i];
            }
        }

        __global__ void kernScatterBit(int n, int* odata, const int* idata,
            const int* bitArray, const int* falseScan, int totalFalse) {
            int i = threadIdx.x + blockIdx.x * blockDim.x;
            if (i < n) {
                int dest = bitArray[i] == 0 ? falseScan[i] : totalFalse + i - falseScan[i];
                odata[dest] = idata[i];
            }
        }

        void sort(int n, int* odata, const int* idata) {
            int* dev_in, * dev_out, * dev_bits, * dev_notBits, * dev_scan;
            int blockSize = 128;
            int numBlocks = (n + blockSize - 1) / blockSize;

            cudaMalloc(&dev_in, n * sizeof(int));
            cudaMalloc(&dev_out, n * sizeof(int));
            cudaMalloc(&dev_bits, n * sizeof(int));
            cudaMalloc(&dev_notBits, n * sizeof(int));
            cudaMalloc(&dev_scan, n * sizeof(int));

            cudaMemcpy(dev_in, idata, n * sizeof(int), cudaMemcpyHostToDevice);

            for (int bit = 0; bit < 32; ++bit) {
                kernExtractBit << <numBlocks, blockSize >> > (n, bit, dev_bits, dev_in);
                kernInvert << <numBlocks, blockSize >> > (n, dev_notBits, dev_bits);
                StreamCompaction::Efficient::scan(n, dev_scan, dev_notBits);

                int totalFalse;
                int lastBool;
                cudaMemcpy(&totalFalse, dev_scan + n - 1, sizeof(int), cudaMemcpyDeviceToHost);
                cudaMemcpy(&lastBool, dev_notBits + n - 1, sizeof(int), cudaMemcpyDeviceToHost);
                totalFalse += lastBool;

                kernScatterBit << <numBlocks, blockSize >> > (n, dev_out, dev_in,
                    dev_bits, dev_scan, totalFalse);

                std::swap(dev_in, dev_out);  // next iteration reads from new sorted array
            }

            cudaMemcpy(odata, dev_in, n * sizeof(int), cudaMemcpyDeviceToHost);

            cudaFree(dev_in);
            cudaFree(dev_out);
            cudaFree(dev_bits);
            cudaFree(dev_notBits);
            cudaFree(dev_scan);
        }
    }
}

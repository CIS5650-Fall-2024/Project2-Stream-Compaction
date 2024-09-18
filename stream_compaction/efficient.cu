#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

namespace StreamCompaction {
    namespace Efficient {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        __global__ void UpSweepAtDepthD(int n, int offset1, int offset2, int* buffer) {
            int k = (blockIdx.x * blockDim.x) + threadIdx.x;
            int offsetBetweenMains = offset1;
            int actualOffset = offset2;

            int index = (k + 1) * offsetBetweenMains - 1;
            if (k < n / offsetBetweenMains) {
                buffer[index] += buffer[index - actualOffset];
            }
        }

        __global__ void DownSweepAtDepthD(int n, int offset1, int offset2, int* buffer) {
            int k = (blockIdx.x * blockDim.x) + threadIdx.x;
            int offsetBetweenMains = offset1;
            int actualOffset = offset2;

            int index = (k + 1) * offsetBetweenMains - 1;
            if (k < n / offsetBetweenMains) {
                //left child index
                int leftChildIndex = index - actualOffset;
                int rightChildIndex = index;
                int leftChildSave = buffer[leftChildIndex];
                buffer[leftChildIndex] = buffer[index];
                buffer[rightChildIndex] = leftChildSave + buffer[index];
            }
        }
        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            int* buffer;
            int size = pow(2, ilog2ceil(n));

            cudaMalloc((void**)&buffer, size * sizeof(int));
            cudaMemset(buffer, 0, size * sizeof(int));
            cudaMemcpy(buffer, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            timer().startGpuTimer();
            // TODO

            //UpSweep (parallel reduction)
            for (int d = 0; d < ilog2ceil(size); d++) {
                dim3 fullBlocksPerGrid((size + blockSize - 1) / blockSize);
                int offsetBetweenMains = 1 << (d + 1);
                int actualOffset = 1 << d;
                UpSweepAtDepthD<<<fullBlocksPerGrid, blockSize>>>(size, offsetBetweenMains, actualOffset, buffer);
            }

            //DownSweep
            cudaMemset(buffer + (size - 1), 0, sizeof(int));
            for (int d = ilog2ceil(n) - 1; d >= 0; d--) {
                dim3 fullBlocksPerGrid((size + blockSize - 1) / blockSize);
                int offsetBetweenMains = 1 << (d + 1);
                int actualOffset = 1 << d;
                DownSweepAtDepthD<<<fullBlocksPerGrid, blockSize>>>(size, offsetBetweenMains, actualOffset, buffer);
                checkCUDAError("DownSweepAtDepthD failed!");
            }

            timer().endGpuTimer();
            //Cpy data back to CPU (only need first n ints! We dont care about the extension anymore)
            cudaMemcpy(odata, buffer, n * sizeof(int), cudaMemcpyDeviceToHost);
            cudaFree(buffer);
        }



        __global__ void KernInitializeBitArray(int n, int* bitArray) {
            int index = (blockIdx.x * blockDim.x) + threadIdx.x;
            if (index < n) {
                if (bitArray[index] == 0) {
                    bitArray[index] = 0;
                }
                else {
                    bitArray[index] = 1;
                }
            }
        }

        __global__ void KernScatter(int n, int* idata, int* scan, int* output) {
            int index = (blockIdx.x * blockDim.x) + threadIdx.x;
            if (index < n) {
                if (idata[index] != 0) {
                    output[scan[index]] = idata[index];
                }
            }
        }

        /**
         * Performs stream compaction on idata, storing the result into odata.
         * All zeroes are discarded.
         *
         * @param n      The number of elements in idata.
         * @param odata  The array into which to store elements.
         * @param idata  The array of elements to compact.
         * @returns      The number of elements remaining after compaction.
         */
        int compact(int n, int *odata, const int *idata) {
            int* numElem = new int;
            *numElem = -1;
            int* bitArray;
            int* idataCpy;
            int* deviceOutput;

            cudaMalloc((void**)&bitArray, n * sizeof(int));
            cudaMalloc((void**)&idataCpy, n * sizeof(int));
            cudaMalloc((void**)&deviceOutput, n * sizeof(int));
            cudaMemset(deviceOutput, 0, n * sizeof(int));
            cudaMemcpy(bitArray, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            cudaMemcpy(idataCpy, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            {
                //PT1: BITARRAY
                // turn buffer1 into a bitarray
                dim3 fullBlocksPerGrid((n + blockSize - 1) / blockSize);
                KernInitializeBitArray<<<fullBlocksPerGrid, blockSize >>>(n, bitArray);
            }

            int* buffer;
            int size = pow(2, ilog2ceil(n));
            {
                //PT2: Scan Initialization
                //buffer is a pow2 size buffer. We copy over buffer1 to this buffer to set it up.
                cudaMalloc((void**)&buffer, size * sizeof(int));
                cudaMemset(buffer, 0, size * sizeof(int));
                cudaMemcpy(buffer, bitArray, n * sizeof(int), cudaMemcpyDeviceToDevice);
            }

            timer().startGpuTimer();
            //TODO
            {
                //PT3: SCAN
                // Store scan result in buffer2
                //UpSweep (parallel reduction)
                for (int d = 0; d < ilog2ceil(size); d++) {
                    dim3 fullBlocksPerGrid((size + blockSize - 1) / blockSize);
                    int offsetBetweenMains = 1 << (d + 1);
                    int actualOffset = 1 << d;
                    UpSweepAtDepthD<<<fullBlocksPerGrid, blockSize>>>(size, offsetBetweenMains, actualOffset, buffer);
                }

                //DownSweep
                cudaMemset(buffer + (size - 1), 0, sizeof(int));
                for (int d = ilog2ceil(n) - 1; d >= 0; d--) {
                    dim3 fullBlocksPerGrid((size + blockSize - 1) / blockSize);
                    int offsetBetweenMains = 1 << (d + 1);
                    int actualOffset = 1 << d;
                    DownSweepAtDepthD<<<fullBlocksPerGrid, blockSize>>>(size, offsetBetweenMains, actualOffset, buffer);
                    checkCUDAError("DownSweepAtDepthD failed!");
                }
            }
            {
                //PT4: SCATTER
                // bitarray: stores bitarray
                // buffer: first n elements store scan
                dim3 fullBlocksPerGrid((n + blockSize - 1) / blockSize);
                KernScatter<<<fullBlocksPerGrid, blockSize>>>(n, idataCpy, buffer, deviceOutput);
            }

            timer().endGpuTimer();

            cudaMemcpy(odata, deviceOutput, n * sizeof(int), cudaMemcpyDeviceToHost);
            //Access final element of scan for numElem
            cudaMemcpy(numElem, buffer + size - 1, sizeof(int), cudaMemcpyDeviceToHost);
            //FREEING
            {
                //compaction free
                cudaFree(buffer);
                cudaFree(bitArray);
                cudaFree(idataCpy);
                cudaFree(deviceOutput);
            }

            return *numElem;

        }
    }
}

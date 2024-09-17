#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"
#include "device_launch_parameters.h"

namespace StreamCompaction {
    namespace Efficient {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        __global__ void kernUpSweep(int n, int d, int* buffer) {
            int index = threadIdx.x + blockIdx.x * blockDim.x;
   
            int pow2tod = 1 << d;
            int pow2todp1 = 2 * pow2tod;

            if (index > n / pow2todp1 - 1) return;
            index *= pow2todp1;

            buffer[index + pow2todp1 - 1] += buffer[index + pow2tod - 1];
        }

        __global__ void kernDownSweep(int n, int d, int s, int* buffer) {
            int index = threadIdx.x + blockIdx.x * blockDim.x;

            int pow2tod = 1 << d;
            int pow2todp1 = 2 * pow2tod;

            if (s) {
                buffer[pow2todp1 - 1] = 0;
            }

            if (index > n / pow2todp1 - 1) return;
            index *= pow2todp1;

            int tmp = buffer[index + pow2tod - 1];
            buffer[index + pow2tod - 1] = buffer[index + pow2todp1 - 1];
            buffer[index + pow2todp1 - 1] += tmp;
        }

        __global__ void kernZeroPadding(int n, int d, int* buffer) {
            int index = threadIdx.x + blockIdx.x * blockDim.x;

            if (index >= 1 << (d + 1) - n) return;

            buffer[n + index] = 0;
        }

        dim3 computeBlocksPerGrid(int threads, int blockSize) {
            return dim3{ (unsigned int)(threads + blockSize - 1) / blockSize };
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            int blockSize = 128;

            bool isPower2Length = (n == (1 << ilog2(n)));

            int bufferLength = (isPower2Length) ? n : 1 << ilog2ceil(n);

            int* tmpArray;
            cudaMalloc((void**)&tmpArray, bufferLength * sizeof(int));
            checkCUDAError("cudaMalloc tmpArray failed!");

            if (!isPower2Length) {
                dim3 blocks = computeBlocksPerGrid(n - (1 << ilog2(n)), blockSize);
                kernZeroPadding<<<blocks, blockSize>>>(n, ilog2(n), tmpArray);
                checkCUDAError("kernZeroPadding failed!");
                cudaDeviceSynchronize();
            }

            cudaMemcpy(tmpArray, idata, n * sizeof(int), cudaMemcpyHostToDevice);

            timer().startGpuTimer();
            // TODO
            for (int d = 0; d < ilog2ceil(n); ++d) {
                dim3 blocks = computeBlocksPerGrid(bufferLength / (1 << (d + 1)), blockSize);
                kernUpSweep<<<blocks, blockSize>>>(bufferLength, d, tmpArray);
                checkCUDAError("kernUpSweep failed!");
                cudaDeviceSynchronize();
            }
            
            bool flag = 1;
            for (int d = ilog2ceil(n) - 1; d >= 0; --d) {
                dim3 blocks = computeBlocksPerGrid(bufferLength / (1 << (d + 1)), blockSize);
                kernDownSweep<<<blocks, blockSize>>>(bufferLength, d, flag, tmpArray);
                flag = 0;
                checkCUDAError("kernDownSweep failed!");
                cudaDeviceSynchronize();
            }
            timer().endGpuTimer();

            cudaMemcpy(odata, tmpArray, n * sizeof(int), cudaMemcpyDeviceToHost);

            cudaFree(tmpArray);
        }

        void scanUntimed(int n, int* odata, const int* idata) {
            int blockSize = 128;

            bool isPower2Length = (n == (1 << ilog2(n)));

            int bufferLength = (isPower2Length) ? n : 1 << ilog2ceil(n);

            int* tmpArray;
            cudaMalloc((void**)&tmpArray, bufferLength * sizeof(int));
            checkCUDAError("cudaMalloc tmpArray failed!");

            if (!isPower2Length) {
                dim3 blocks = computeBlocksPerGrid(n - (1 << ilog2(n)), blockSize);
                kernZeroPadding << <blocks, blockSize >> > (n, ilog2(n), tmpArray);
                checkCUDAError("kernZeroPadding failed!");
                cudaDeviceSynchronize();
            }

            cudaMemcpy(tmpArray, idata, n * sizeof(int), cudaMemcpyHostToDevice);

            // TODO
            for (int d = 0; d < ilog2ceil(n); ++d) {
                dim3 blocks = computeBlocksPerGrid(bufferLength / (1 << (d + 1)), blockSize);
                kernUpSweep<<<blocks, blockSize>>>(bufferLength, d, tmpArray);
                checkCUDAError("kernUpSweep failed!");
                cudaDeviceSynchronize();
            }

            bool flag = 1;
            for (int d = ilog2ceil(n) - 1; d >= 0; --d) {
                dim3 blocks = computeBlocksPerGrid(bufferLength / (1 << (d + 1)), blockSize);
                kernDownSweep<<<blocks, blockSize>>>(bufferLength, d, flag, tmpArray);
                flag = 0;
                checkCUDAError("kernDownSweep failed!");
                cudaDeviceSynchronize();
            }

            cudaMemcpy(odata, tmpArray, n * sizeof(int), cudaMemcpyDeviceToHost);

            cudaFree(tmpArray);
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
        int compact(int n, int* odata, const int* idata) {
            int blockSize = 128;
            dim3 blocks{ (unsigned int)(n + blockSize - 1) / blockSize };

            int* dev_buffer1;
            int* dev_buffer2;
            int* dev_boolArray;
            int* dev_indices;
            cudaMalloc((void**)&dev_boolArray, n * sizeof(int));
            checkCUDAError("cudaMalloc dev_boolArray failed!");
            cudaMalloc((void**)&dev_indices,   n * sizeof(int));
            cudaDeviceSynchronize();
            checkCUDAError("cudaMalloc dev_indices failed!");
            cudaMalloc((void**)&dev_buffer1,   n * sizeof(int));
            checkCUDAError("cudaMalloc dev_buffer1 failed!");
            cudaMalloc((void**)&dev_buffer2,   n * sizeof(int));
            checkCUDAError("cudaMalloc dev_buffer2 failed!");

            cudaMemcpy(dev_buffer1, idata, n * sizeof(int), cudaMemcpyHostToDevice);

            timer().startGpuTimer();
            cudaDeviceSynchronize();
            checkCUDAError("timer failed!");
            // TODO
            

            StreamCompaction::Common::kernMapToBoolean<<<blocks, blockSize>>>(n, dev_boolArray, dev_buffer1);
            cudaDeviceSynchronize();
            checkCUDAError("kernMapToBoolean failed!");
            
            scanUntimed(n, dev_indices, dev_boolArray);

            StreamCompaction::Common::kernScatter<<<blocks, blockSize>>>(n, dev_buffer2, dev_buffer1, dev_boolArray, dev_indices);
            checkCUDAError("kernScatter failed!");
            cudaDeviceSynchronize();
            
            cudaMemcpy(odata, dev_buffer2, n * sizeof(int), cudaMemcpyDeviceToHost);
            
            int numElem;
            cudaMemcpy(&numElem, &dev_indices[n - 1], sizeof(int), cudaMemcpyDeviceToHost);
            
            timer().endGpuTimer();

            cudaFree(dev_boolArray);
            cudaFree(dev_indices);
            cudaFree(dev_buffer1);
            cudaFree(dev_buffer2);
            return numElem;
        }
    }
}
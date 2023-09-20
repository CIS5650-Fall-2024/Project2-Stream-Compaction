#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

#define BlockSize 256
namespace StreamCompaction {
    namespace Efficient {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        __global__ void kernUpSweep(int n, int d, int* odata)
        {
            int index = threadIdx.x + (blockIdx.x * blockDim.x);
            if (index >= n) return;
            int path = 1 << d;
            if (index % (2 * path) == 0)
            {
                odata[index + 2 * path - 1] += odata[index + path - 1];
            }
        }

        __global__ void kernDownSweep(int n, int d, int* odata)
        {
            int index = threadIdx.x + (blockIdx.x * blockDim.x);
            if (index >= n) return;
            int path = 1 << d;
            if (index % (2 * path) == 0)
            {
                int temp = odata[index + path - 1];
                odata[index + path - 1] = odata[index + 2 * path - 1];
                odata[index + 2 * path - 1] += temp;
            }
        }

        __global__ void kernUpSweepOptimized(int n, int d, int* odata) {
            int path = 1 << d;
            int index = (threadIdx.x + blockIdx.x * blockDim.x) * 2 * path;

            if (index + 2 * path - 1 < n) {
                odata[index + 2 * path - 1] += odata[index + path - 1];
            }
        }

        __global__ void kernDownSweepOptimized(int n, int d, int* odata) {
            int path = 1 << d;
            int index = (threadIdx.x + blockIdx.x * blockDim.x) * 2 * path;

            if (index + 2 * path - 1 < n) {
                int temp = odata[index + path - 1];
                odata[index + path - 1] = odata[index + 2 * path - 1];
                odata[index + 2 * path - 1] += temp;
            }
        }

        void scanUpDownSweep(int n, int* odata)
        {
            dim3 BlockDim((BlockSize + n - 1) / BlockSize);
            for (int d = 0; d <= ilog2ceil(n) - 1; ++d) {
                kernUpSweep <<< BlockDim, BlockSize >>> (n, d, odata);
            }
            cudaDeviceSynchronize();

            cudaMemsetAsync(odata + n - 1, 0, sizeof(int));

            for (int d = ilog2ceil(n) - 1; d >= 0; --d) {
                kernDownSweep <<< BlockDim, BlockSize >>> (n, d, odata);
            }
            cudaDeviceSynchronize();
        }

        void scanUpDownSweepOptimized(int n, int* odata) {
            for (int d = 0; d <= ilog2ceil(n) - 1; ++d) {
                int threadsNeeded = n / (2 << d);
                dim3 BlockDim = (threadsNeeded + BlockSize - 1) / BlockSize;

                kernUpSweepOptimized << < BlockDim, BlockSize >> > (n, d, odata);
            }
            cudaDeviceSynchronize();

            cudaMemsetAsync(odata + n - 1, 0, sizeof(int));

            for (int d = ilog2ceil(n) - 1; d >= 0; --d) {
                int threadsNeeded = n / (2 << d);
                dim3 BlockDim = (threadsNeeded + BlockSize - 1) / BlockSize;

                kernDownSweepOptimized <<< BlockDim, BlockSize >>> (n, d, odata);
            }
            cudaDeviceSynchronize();
        }



        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            int* dev_data;
            int arrSize = 1 << ilog2ceil(n);
            cudaMalloc((void**)&dev_data, arrSize * sizeof(int));
            cudaMemcpy(dev_data, idata, sizeof(int) * n, cudaMemcpyHostToDevice);

            timer().startGpuTimer();

            scanUpDownSweep(arrSize, dev_data);

            timer().endGpuTimer();

            cudaMemcpy(odata, dev_data, sizeof(int) * n, cudaMemcpyDeviceToHost);
            cudaFree(dev_data);
        }

        void scanOptimized(int n, int* odata, const int* idata) {
            int* dev_data;
            int arrSize = 1 << ilog2ceil(n);
            cudaMalloc((void**)&dev_data, arrSize * sizeof(int));
            cudaMemcpy(dev_data, idata, sizeof(int) * n, cudaMemcpyHostToDevice);

            timer().startGpuTimer();

            scanUpDownSweepOptimized(arrSize, dev_data);

            timer().endGpuTimer();

            cudaMemcpy(odata, dev_data, sizeof(int) * n, cudaMemcpyDeviceToHost);
            cudaFree(dev_data);
        }

        void scanForRadixSort(int n, int* odata, const int* idata) {
            int* dev_data;
            int arrSize = 1 << ilog2ceil(n);
            cudaMalloc((void**)&dev_data, arrSize * sizeof(int));
            cudaMemcpy(dev_data, idata, sizeof(int) * n, cudaMemcpyHostToDevice);

            scanUpDownSweep(arrSize, dev_data);

            cudaMemcpy(odata, dev_data, sizeof(int) * n, cudaMemcpyDeviceToHost);
            cudaFree(dev_data);
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
            // TODO
            int arrSize = 1 << ilog2ceil(n);
            int* dev_bools;
            int* dev_indices;
            int* dev_idata;
            int* dev_odata;

            cudaMalloc((void**)&dev_bools, n * sizeof(int));
            cudaMalloc((void**)&dev_indices, arrSize * sizeof(int));
            cudaMalloc((void**)&dev_idata, n * sizeof(int));
            cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);

            timer().startGpuTimer();
            // 1. Map the input data to a boolean array
            dim3 fullBlocksPerGrid = (n + BlockSize - 1) / BlockSize;

            StreamCompaction::Common::kernMapToBoolean <<<fullBlocksPerGrid, BlockSize >>> (n, dev_bools, dev_idata);
            cudaDeviceSynchronize();

            // 2. Compute the exclusive prefix sum on the boolean array
            cudaMemcpy(dev_indices, dev_bools, n * sizeof(int), cudaMemcpyDeviceToDevice);
            cudaMemset(dev_indices + n, 0, (arrSize - n) * sizeof(int));
            scanUpDownSweep(arrSize, dev_indices);

            // The total number of elements after compaction is stored in the last slot of indices + the last slot of bools
            int totalSize;
            int lastBool, lastIndex;
            cudaMemcpy(&lastBool, &dev_bools[n - 1], sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(&lastIndex, &dev_indices[n - 1], sizeof(int), cudaMemcpyDeviceToHost);
            totalSize = lastBool + lastIndex;

            cudaMalloc((void**)&dev_odata, totalSize * sizeof(int));
            // 3. Scatter the data to the output array using the index array
            StreamCompaction::Common::kernScatter <<<fullBlocksPerGrid, BlockSize >>> (n, dev_odata, dev_idata, dev_bools, dev_indices);
            cudaDeviceSynchronize();

            timer().endGpuTimer();
            
            cudaMemcpy(odata, dev_odata, totalSize * sizeof(int), cudaMemcpyDeviceToHost);
            cudaFree(dev_bools);
            cudaFree(dev_indices);
            cudaFree(dev_idata);
            cudaFree(dev_odata);

            return totalSize;
        }

        int compactOptimized(int n, int* odata, const int* idata) {
            // TODO
            int arrSize = 1 << ilog2ceil(n);
            int* dev_bools;
            int* dev_indices;
            int* dev_idata;
            int* dev_odata;

            cudaMalloc((void**)&dev_bools, n * sizeof(int));
            cudaMalloc((void**)&dev_indices, arrSize * sizeof(int));
            cudaMalloc((void**)&dev_idata, n * sizeof(int));
            cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);

            timer().startGpuTimer();
            // 1. Map the input data to a boolean array
            dim3 fullBlocksPerGrid = (n + BlockSize - 1) / BlockSize;

            StreamCompaction::Common::kernMapToBoolean << <fullBlocksPerGrid, BlockSize >> > (n, dev_bools, dev_idata);
            cudaDeviceSynchronize();

            // 2. Compute the exclusive prefix sum on the boolean array
            cudaMemcpy(dev_indices, dev_bools, n * sizeof(int), cudaMemcpyDeviceToDevice);
            cudaMemset(dev_indices + n, 0, (arrSize - n) * sizeof(int));
            scanUpDownSweepOptimized(arrSize, dev_indices);

            // The total number of elements after compaction is stored in the last slot of indices + the last slot of bools
            int totalSize;
            int lastBool, lastIndex;
            cudaMemcpy(&lastBool, &dev_bools[n - 1], sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(&lastIndex, &dev_indices[n - 1], sizeof(int), cudaMemcpyDeviceToHost);
            totalSize = lastBool + lastIndex;

            cudaMalloc((void**)&dev_odata, totalSize * sizeof(int));
            // 3. Scatter the data to the output array using the index array
            StreamCompaction::Common::kernScatter << <fullBlocksPerGrid, BlockSize >> > (n, dev_odata, dev_idata, dev_bools, dev_indices);
            cudaDeviceSynchronize();

            timer().endGpuTimer();

            cudaMemcpy(odata, dev_odata, totalSize * sizeof(int), cudaMemcpyDeviceToHost);
            cudaFree(dev_bools);
            cudaFree(dev_indices);
            cudaFree(dev_idata);
            cudaFree(dev_odata);

            return totalSize;
        }

        __global__ void kernComputeBEArr(int n, int pos, const int* idata, int* bArr, int* eArr)
        {
            int index = blockIdx.x * blockDim.x + threadIdx.x;
            if (index >= n) return;
            int bitVal = (idata[index] >> pos) & 1;
            bArr[index] = bitVal;
            eArr[index] = !bitVal;
        }

        __global__ void kernComputeTArr(int n, int totalFalse, const int* fArr, int* tArr)
        {
            int index = blockIdx.x * blockDim.x + threadIdx.x;
            if (index >= n) return;
            tArr[index] = index - fArr[index] + totalFalse;
        }

        __global__ void kernComputeDArr(int n, const int* bArr, const int* tArr, const int* fArr, int* dArr)
        {
            int index = blockIdx.x * blockDim.x + threadIdx.x;
            if (index >= n) return;
            dArr[index] = bArr[index] ? tArr[index] : fArr[index];
        }

        __global__ void kernScatterOutput(int n, const int* dArr, const int* idata, int* odata)
        {
            int index = blockIdx.x * blockDim.x + threadIdx.x;
            if (index >= n) return;
            odata[dArr[index]] = idata[index];
        }

        void radixSort(int n, int* odata, const int* idata)
        {
            int* dev_idata;
            int* dev_odata;

            int* dev_bArr;
            int* dev_eArr;
            int* dev_fAr;
            int* dev_tArr;
            int* dev_dArr;

            cudaMalloc((void**)&dev_idata, n * sizeof(int));
            cudaMalloc((void**)&dev_bArr, n * sizeof(int));
            cudaMalloc((void**)&dev_eArr, n * sizeof(int));
            cudaMalloc((void**)&dev_fAr, n * sizeof(int));
            cudaMalloc((void**)&dev_tArr, n * sizeof(int));
            cudaMalloc((void**)&dev_dArr, n * sizeof(int));
            cudaMalloc((void**)&dev_odata, n * sizeof(int));

            cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);

            int blockDim = (n + BlockSize - 1) / BlockSize;

            int totalFalse = 0;
            int maxVal = *(std::max_element(idata, idata + n));

            timer().startGpuTimer();
            for (int pos = 0; pos < ilog2ceil(maxVal); pos++) {
                kernComputeBEArr <<<blockDim, BlockSize >>> (n, pos, dev_idata, dev_bArr, dev_eArr);
                //cudaDeviceSynchronize();

                scanForRadixSort(n, dev_fAr, dev_eArr);

                //Compute TotalFalse
                int e, f;
                cudaMemcpy(&e, dev_eArr + n - 1, sizeof(int), cudaMemcpyDeviceToHost);
                cudaMemcpy(&f, dev_fAr + n - 1, sizeof(int), cudaMemcpyDeviceToHost);
                totalFalse = e + f;

                kernComputeTArr <<< blockDim, BlockSize >>> (n, totalFalse, dev_fAr, dev_tArr);
                //cudaDeviceSynchronize();

                kernComputeDArr <<< blockDim, BlockSize >>> (n, dev_bArr, dev_tArr, dev_fAr, dev_dArr);
                //cudaDeviceSynchronize();

                kernScatterOutput <<<blockDim, BlockSize >>> (n, dev_dArr, dev_idata, dev_odata);
                //cudaDeviceSynchronize();
                std::swap(dev_odata, dev_idata);
            }
            timer().endGpuTimer();
            cudaMemcpy(odata, dev_idata, n * sizeof(int), cudaMemcpyDeviceToHost);

            cudaFree(dev_idata);
            cudaFree(dev_odata);
            cudaFree(dev_bArr);
            cudaFree(dev_eArr);
            cudaFree(dev_fAr);
            cudaFree(dev_tArr);
            cudaFree(dev_dArr);
        }

    }
}

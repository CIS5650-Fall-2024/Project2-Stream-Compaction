#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"
/*! Block size used for CUDA kernel launch. */
#define blockSize 128
#define passNumber 6

namespace StreamCompaction {
    namespace Efficient {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        //number of data with padding included
        int length;
        //container to store the data
        int* dev_idata;
        int* dev_odata;
        int* dev_indices;
        int* dev_bools;
        int* dev_nonZero;

        //for radix sort (0 - 5)
        int* dev_input;
        int* dev_pass;

        dim3 threadsPerBlock(blockSize);

        void initRadixSort(int N, int* odata, const int* idata)
        {
            //give padding the arrays with size not the power of 2
            length = int(pow(2, ilog2ceil(N)));

            cudaMalloc((void**)&dev_input, length * sizeof(int));
            cudaMemcpy(dev_input, idata, N * sizeof(int), cudaMemcpyHostToDevice);
            checkCUDAError("cudaMalloc dev_input failed!", __LINE__);

            cudaMalloc((void**)&dev_idata, length * sizeof(int));
            cudaMemcpy(dev_idata, idata, N * sizeof(int), cudaMemcpyHostToDevice);
            checkCUDAError("cudaMalloc dev_idata failed!", __LINE__);

            cudaMalloc((void**)&dev_odata, length * sizeof(int));
            cudaMemcpy(dev_odata, idata, N * sizeof(int), cudaMemcpyHostToDevice);
            checkCUDAError("cudaMalloc dev_odata failed!", __LINE__);

            cudaMalloc((void**)&dev_pass, length * sizeof(int));
            checkCUDAError("cudaMalloc dev_pass failed!", __LINE__);

            //to record non-zero numbers 
            cudaMalloc((void**)&dev_nonZero, 1 * sizeof(int));
            checkCUDAError("cudaMalloc dev_nonZero failed!", __LINE__);

            cudaDeviceSynchronize();
        }

        void initScan(int N, int* odata, const int* idata)
        {
            //give padding the arrays with size not the power of 2
            length = int(pow(2, ilog2ceil(N)));
            cudaMalloc((void**)&dev_idata, length * sizeof(int));
            cudaMemcpy(dev_idata, idata, N * sizeof(int), cudaMemcpyHostToDevice);
            checkCUDAError("cudaMalloc dev_idata failed!", __LINE__);
            cudaMalloc((void**)&dev_odata, length * sizeof(int));
            cudaMemcpy(dev_odata, idata, N * sizeof(int), cudaMemcpyHostToDevice);
            checkCUDAError("cudaMalloc dev_odata failed!", __LINE__);
            //to record non-zero numbers 
            cudaMalloc((void**)&dev_nonZero, 1 * sizeof(int));
            checkCUDAError("cudaMalloc dev_nonZero failed!", __LINE__);
            cudaDeviceSynchronize();
        }

        void initCompact(int N, const int* idata, int* odata)
        {
            //give padding the arrays with size not the power of 2
            length = int(pow(2, ilog2ceil(N)));
            cudaMalloc((void**)&dev_idata, length * sizeof(int));
            cudaMemcpy(dev_idata, idata, N * sizeof(int), cudaMemcpyHostToDevice);
            checkCUDAError("cudaMalloc dev_idata failed!", __LINE__);
            cudaMalloc((void**)&dev_odata, length * sizeof(int));
            checkCUDAError("cudaMalloc dev_odata failed!", __LINE__);
            cudaMalloc((void**)&dev_indices, length * sizeof(int));
            cudaMemcpy(dev_indices, odata, N * sizeof(int), cudaMemcpyHostToDevice);
            checkCUDAError("cudaMalloc dev_indices failed!", __LINE__);
            cudaMalloc((void**)&dev_bools, length * sizeof(int));
            checkCUDAError("cudaMalloc dev_bool failed!", __LINE__);
        }

        void endScan()
        {
            cudaFree(dev_idata);
            cudaFree(dev_odata);
        }

        void endCompact()
        {
            cudaFree(dev_bools);
            cudaFree(dev_indices);
            cudaFree(dev_idata);
            cudaFree(dev_odata);
            cudaFree(dev_nonZero);
        }
        
        void endRadixSort()
        {
            cudaFree(dev_idata);
            cudaFree(dev_odata);
            cudaFree(dev_pass);
            cudaFree(dev_input);
            cudaFree(dev_nonZero);
        }

        //n is number of elements in one pass
        //pass is 0, 1, 2, 3, 4, 5
        __global__ void RadixSortMapKernel(int n, int realLen, int pass, int* idata, int* odata)
        {
            int index = (blockIdx.x * blockDim.x) + threadIdx.x;
            if (index >= n)
            {
                return;
            }
            //map 0 to 1
            odata[index] = 1 - ((idata[index] >> pass) & 1);
            //if there are extra space mark them as 1
            if (index >= realLen)
            {
                odata[index] = 0;
            }
        }

        //here n is the number of the array without padding
        __global__ void RadixTrueScanKernel(int n, int zeros, int* idata, int* pass, int* odata)
        {
            int index = (blockIdx.x * blockDim.x) + threadIdx.x;
            if (index >= n)
            {
                return;
            }
            if (pass[index] == 0)
            {
                odata[index] = index - idata[index] + zeros;
            }
            else
            {
                odata[index] = idata[index];
            }
        }

        
        __global__ void RadixScatterKernel(int n, int* indices, int* odata, int* input)
        {
            int index = (blockIdx.x * blockDim.x) + threadIdx.x;
            if (index >= n)
            {
                return;
            }
            odata[indices[index]] = input[index];
        }

        __global__ void EfficientMapKernel(int n, int realLen, int* idata)
        {
            int index = (blockIdx.x * blockDim.x) + threadIdx.x;
            if (index >= n)
            {
                return;
            }

            if (index < realLen && idata[index] != 0)
            {
                idata[index] = 1;
            }
            else
            {
                idata[index] = 0;
            }
            //__syncthreads();
        }

        __global__ void InitDownSweepKernel(int n, int* idata, int* odata)
        {
            idata[n - 1] = 0;
            odata[n - 1] = 0;
        }

        __global__ void UpSweepScanKernel(int n, int interval, int* idata, int* odata)
        {
            int index = (blockIdx.x * blockDim.x) + threadIdx.x;
            if (index >= n)
            {
                return;
            }
            //index that needs to be changed
            int validBase = interval * 2;
            int trueIndex = validBase * (index + 1) - 1;
            idata[trueIndex] = odata[trueIndex - interval] + odata[trueIndex];
            odata[trueIndex] = idata[trueIndex];
        }

        __global__ void DownSweepScanKernel(int n, int interval, int* idata, int* odata)
        {
            int index = (blockIdx.x * blockDim.x) + threadIdx.x;
            if (index >= n)
            {
                return;
            }
            //index that needs to be changed
            int validBase = interval * 2;
            int trueIndex = validBase * (index + 1) - 1;
            idata[trueIndex] = odata[trueIndex - interval] + odata[trueIndex];
            idata[trueIndex - interval] = odata[trueIndex];
        }

        __global__ void CountKernel(int n, int* indices, int* idata, int* count)
        {
            int lastIndice = n - 1;
            if (idata[lastIndice] == 0)
            {
                count[0] = indices[lastIndice];
            }
            else
            {
                count[0] = indices[lastIndice] + 1;
            }
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            // TODO
            initScan(n, odata, idata);
            dim3 fullBlocksPerGrid((length + blockSize - 1) / blockSize);
            timer().startGpuTimer();
            for (int i = 1; i < length; i = i * 2)
            {
                int totalSize = length / (2 * i);
                dim3 BlocksPerGrid((totalSize + blockSize - 1) / blockSize);
                UpSweepScanKernel << <BlocksPerGrid, blockSize >> > (totalSize, i, dev_idata, dev_odata);
                cudaDeviceSynchronize();

                std::swap(dev_idata, dev_odata);
            }
            std::swap(dev_idata, dev_odata);
            InitDownSweepKernel << <1, 1 >> > (length, dev_idata, dev_odata);

            for (int i = length / 2; i > 0; i = i / 2)
            {
                int totalSize = length / (2 * i);
                dim3 BlocksPerGrid((totalSize + blockSize - 1) / blockSize);
                DownSweepScanKernel << <BlocksPerGrid, blockSize >> > (totalSize , i, dev_idata, dev_odata);
                cudaDeviceSynchronize();
                std::swap(dev_idata, dev_odata);
            }
            timer().endGpuTimer();
            cudaMemcpy(odata, dev_odata, sizeof(int) * n, cudaMemcpyDeviceToHost);

            endScan();
        }

        void printArray(int n, int* a, bool abridged = false) {
            printf("    [ ");
            for (int i = 0; i < n; i++) {
                if (abridged && i + 2 == 15 && n > 16) {
                    i = n - 2;
                    printf("... ");
                }
                printf("%3d ", a[i]);
            }
            printf("]\n");
        }

        void radixSort(int n, int* odata, const int* idata)
        {
            initRadixSort(n, odata, idata);
            dim3 fullBlocksPerGrid((length + blockSize - 1) / blockSize);
            for (int i = 0; i < passNumber; i++)
            {

                RadixSortMapKernel << <fullBlocksPerGrid, blockSize >> > (length, n, i, dev_input, dev_odata);
                cudaMemcpy(dev_pass, dev_odata, sizeof(int) * length, cudaMemcpyDeviceToDevice);
                /*
                * Scan for one pass begins
                */
                for (int i = 1; i < length; i = i * 2)
                {
                    int totalSize = length / (2 * i);
                    dim3 BlocksPerGrid((totalSize + blockSize - 1) / blockSize);
                    UpSweepScanKernel << <BlocksPerGrid, blockSize >> > (totalSize, i, dev_idata, dev_odata);
                    cudaDeviceSynchronize();

                    std::swap(dev_idata, dev_odata);
                }
                std::swap(dev_idata, dev_odata);
                InitDownSweepKernel << <1, 1 >> > (length, dev_idata, dev_odata);

                for (int i = length / 2; i > 0; i = i / 2)
                {
                    int totalSize = length / (2 * i);
                    dim3 BlocksPerGrid((totalSize + blockSize - 1) / blockSize);
                    DownSweepScanKernel << <BlocksPerGrid, blockSize >> > (totalSize, i, dev_idata, dev_odata);
                    cudaDeviceSynchronize();
                    std::swap(dev_idata, dev_odata);
                }
                /*
                * scan for one pass ends
                */
                //get the total number of zero
                int* p = (int*)malloc(1 * sizeof(int));
                CountKernel << <1, 1 >> > (n, dev_odata, dev_pass, dev_nonZero);
                cudaMemcpy(p, dev_nonZero, sizeof(int) * 1, cudaMemcpyDeviceToHost);
                int zeros = p[0];

                dim3 fullBlocksPerGrid1((n + blockSize - 1) / blockSize);
                RadixTrueScanKernel << <fullBlocksPerGrid1, blockSize >> > (n, zeros, dev_odata, dev_pass, dev_idata);
                RadixScatterKernel<< <fullBlocksPerGrid1, blockSize >> >(n, dev_idata, dev_odata, dev_input);
                std::swap(dev_odata, dev_input);
            }
            //dev_input as the final result
            cudaMemcpy(odata, dev_input, sizeof(int) * n, cudaMemcpyDeviceToHost);
            endRadixSort();
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

            timer().startGpuTimer();

            initScan(n, odata, idata);
            dim3 fullBlocksPerGrid((length + blockSize - 1) / blockSize);
            EfficientMapKernel << <fullBlocksPerGrid, blockSize >> > (length, n, dev_odata);
            for (int i = 1; i < length; i = i * 2)
            {
                int totalSize = length / (2 * i);
                dim3 BlocksPerGrid((totalSize + blockSize - 1) / blockSize);
                UpSweepScanKernel << <BlocksPerGrid, blockSize >> > (totalSize, i, dev_idata, dev_odata);
                cudaDeviceSynchronize();

                std::swap(dev_idata, dev_odata);
            }
            std::swap(dev_idata, dev_odata);
            InitDownSweepKernel << <1, 1 >> > (length, dev_idata, dev_odata);

            for (int i = length / 2; i > 0; i = i / 2)
            {
                int totalSize = length / (2 * i);
                dim3 BlocksPerGrid((totalSize + blockSize - 1) / blockSize);
                DownSweepScanKernel << <BlocksPerGrid, blockSize >> > (totalSize, i, dev_idata, dev_odata);
                cudaDeviceSynchronize();
                std::swap(dev_idata, dev_odata);
            }
            cudaMemcpy(odata, dev_odata, sizeof(int) * n, cudaMemcpyDeviceToHost);

            endScan();

            initCompact(n, idata, odata);
            StreamCompaction::Common::kernMapToBoolean << <fullBlocksPerGrid, blockSize >> > (length, dev_bools, dev_indices);
            StreamCompaction::Common::kernScatter << <fullBlocksPerGrid, blockSize >> > (length, dev_odata, dev_idata, dev_bools, dev_indices);
            cudaMemcpy(odata, dev_odata, sizeof(int) * n, cudaMemcpyDeviceToHost);
            int* p = (int*)malloc(1 * sizeof(int));
            CountKernel << <1, 1 >> > (n, dev_indices, dev_idata, dev_nonZero);
            cudaMemcpy(p, dev_nonZero, sizeof(int) * 1, cudaMemcpyDeviceToHost);
            int ans = p[0];
            free(p);
            endCompact();
            timer().endGpuTimer();
            return ans;
        }
    }
}

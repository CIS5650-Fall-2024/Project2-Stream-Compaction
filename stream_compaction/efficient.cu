#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"
#include <iostream>

namespace StreamCompaction {
    namespace Efficient {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        __global__ void initData(int n, int max, int* data) {
            int index = (blockIdx.x * blockDim.x) + threadIdx.x;
            if (index < (max - n)) {
                data[n + index] = 0;
            }
        }

        __global__ void changeNum(int i, int newNum, int* data) {
            int index = (blockIdx.x * blockDim.x) + threadIdx.x;
            if (index == 0) {
                data[i] = newNum;
            }
        }
        
        __global__ void upSweep(int N, int offsetBase, int* data) {
            int index = (blockIdx.x * blockDim.x) + threadIdx.x;
            int offset = 1 << offsetBase;
            if (index * offset * 2 - 1 < N) {
                data[index * offset * 2 - 1] += data[index * offset * 2 - offset - 1];
            }
        }

        __global__ void downSweep(int N, int offsetBase, int* data) {
            int index = (blockIdx.x * blockDim.x) + threadIdx.x;
            int offset = 1 << offsetBase;
            if (index * offset * 2 - 1 < N) {
                int t = data[index * offset * 2 - offset - 1];
                data[index * offset * 2 - offset - 1] = data[index * offset * 2 - 1];
                data[index * offset * 2 - 1] += t;
            }
        }

       

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            
            const int blockSize = 32;
            const int max = ((n - 1) / blockSize + 1) * blockSize;

            std::cout << "n = " << n << ", max = " << max << std::endl;

            int* dev_data;
            cudaMalloc((void**)&dev_data, max * sizeof(int));
            checkCUDAError("cudaMalloc dev_data failed!");
            cudaMemcpy(dev_data, idata, sizeof(int) * n, cudaMemcpyHostToDevice);
            checkCUDAError("cudaMemcpy idata -> dev_data failed!");

            timer().startGpuTimer();
            //if (max > n) {
            //    dim3 initBlockNum((max - n + blockSize - 1) / blockSize);
            //    initData << <initBlockNum, blockSize >> > (n, max, dev_data);
            //}

            // up sweep
            int addTimes = max / 2;
            for (int i = 0; i < ilog2ceil(max); i++) {
                dim3 fullBlocksPerGrid((addTimes + blockSize) / blockSize);
                upSweep << <fullBlocksPerGrid, blockSize >> > (max, i, dev_data);
                addTimes /= 2;
            }

            // down sweep
            int swapTime = 1;
            changeNum << <1, 1 >> > (max - 1, 0, dev_data);

            for (int i = ilog2ceil(max) - 1; i >= 0; i--) {
                dim3 fullBlocksPerGrid((swapTime + blockSize) / blockSize);
                downSweep << <fullBlocksPerGrid, blockSize >> > (max, i, dev_data);
                swapTime *= 2;
            }

            //cudaMemcpy(odata, dev_data, sizeof(int) * n, cudaMemcpyDeviceToHost);
            //checkCUDAError("cudaMemcpy dev_data -> odata for cout failed!");
            //for (int i = 0; i < n; i++) { std::cout << odata[i] << ", "; }
            //std::cout << std::endl << std::endl << std::endl << std::endl;

            timer().endGpuTimer();
            cudaMemcpy(odata, dev_data, sizeof(int) * n, cudaMemcpyDeviceToHost);
            checkCUDAError("cudaMemcpy dev_data -> odata failed!");

            cudaFree(dev_data);
            checkCUDAError("cudaFree failed!");
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
            // TODO
            timer().endGpuTimer();
            return -1;
        }
    }
}

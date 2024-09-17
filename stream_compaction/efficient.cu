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

        __global__ void kernelEfficientScanUpSweep(int N, int d, int* data) {
            int index = threadIdx.x + (blockIdx.x * blockDim.x);
            int d_2 = 1 << d;
            if (index < N && ((index + 1) % (d_2 * 2) == 0)) {
                data[index] += data[index - d_2];
            }
        }

        __global__ void kernelEfficientScanDownSweep(int N, int d, int* data) {
            int index = threadIdx.x + (blockIdx.x * blockDim.x);
            int d_2 = 1 << d;
            if (index < N && ((index + 1) % (d_2 * 2) == 0)) {
                int tmp = data[index];
                data[index] += data[index - d_2];
                data[index - d_2] = tmp;
            }
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            // iteration initialization
            int iterNum = ilog2ceil(n);
            int tmpLength = 1 << iterNum;
            int blockNum((tmpLength + blockSize - 1) / blockSize);

            //device memory initialized
            int* dev_tmp;
            cudaMalloc((void**)&dev_tmp, tmpLength * sizeof(int));
            checkCUDAError("cudaMalloc dev_tmp failed!");  
            // copy array from cpu to gpu
            cudaMemcpy(dev_tmp, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            checkCUDAError("cudaMemcpy idata to dev_tmp failed!");
            // fill pad with 0
            if (tmpLength > n) {
                cudaMemset(dev_tmp + n, 0, (tmpLength - n) * sizeof(int));
                checkCUDAError("cudaMemset dev_tmp pad values failed!");
            }

            timer().startGpuTimer();
            
            // up sweep
            for (int d = 0; d < iterNum; d++) {
                kernelEfficientScanUpSweep<<<blockNum, blockSize>>>(tmpLength, d, dev_tmp);
                checkCUDAError("kernelEfficientScanUpSweep failed!");
            }
            cudaDeviceSynchronize();

            //set root to 0
            cudaMemset(dev_tmp + tmpLength - 1, 0, sizeof(int));
            checkCUDAError("cudaMemset dev_tmp root to 0 failed!");

            // down sweep
            for (int d = iterNum - 1; d >= 0; d--) {
                kernelEfficientScanDownSweep<<<blockNum, blockSize>>>(tmpLength, d, dev_tmp);
                checkCUDAError("kernelEfficientScanDownSweep failed!");
            }

            timer().endGpuTimer();

            // copy array from gpu to cpu
            cudaMemcpy(odata, dev_tmp, n * sizeof(int), cudaMemcpyDeviceToHost);
            checkCUDAError("cudaMemcpy final odata failed!");

            // free memory
            cudaFree(dev_tmp);
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
            // iteration initialization
            int iterNum = ilog2ceil(n);
            int tmpLength = 1 << iterNum;
            int originalBlockNum((n + blockSize - 1) / blockSize);
            int blockNum((tmpLength + blockSize - 1) / blockSize);

            //device memory initialized
            int* dev_idata;
            int* dev_bools;
            int* dev_indices;
            int* dev_odata;
            cudaMalloc((void**)&dev_idata, n * sizeof(int));
            checkCUDAError("cudaMalloc dev_idate failed!");
            cudaMalloc((void**)&dev_odata, n * sizeof(int));
            checkCUDAError("cudaMalloc dev_odata failed!");
            cudaMalloc((void**)&dev_bools, n * sizeof(int));
            checkCUDAError("cudaMalloc dev_bools failed!");
            cudaMalloc((void**)&dev_indices, tmpLength * sizeof(int));
            checkCUDAError("cudaMalloc dev_indices failed!");
            // copy array from cpu to gpu
            cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            checkCUDAError("cudaMemcpy idata to dev_idate failed!");
            
            timer().startGpuTimer();
            
            // map
            Common::kernMapToBoolean<<<originalBlockNum, blockSize>>>(n, dev_bools, dev_idata);
            checkCUDAError("kernMapToBoolean failed!");

            //copy from dev_bools to dev_indices for scan
            cudaMemcpy(dev_indices, dev_bools, n * sizeof(int), cudaMemcpyDeviceToDevice);
            checkCUDAError("cudaMemcpy dev_bools to dev_indices failed!");

            // fill pad with 0
            if (tmpLength > n) {
                cudaMemset(dev_indices + n, 0, (tmpLength - n) * sizeof(int));
                checkCUDAError("cudaMemset dev_indices pad values failed!");
            }

            // up sweep
            for (int d = 0; d < iterNum; d++) {
                kernelEfficientScanUpSweep<<<blockNum, blockSize>>>(tmpLength, d, dev_indices);
                checkCUDAError("kernelEfficientScanUpSweep failed!");
            }
            cudaDeviceSynchronize();

            //set root to 0
            cudaMemset(dev_indices + tmpLength - 1, 0, sizeof(int));
            checkCUDAError("cudaMemset dev_indices root to 0 failed!");

            // down sweep
            for (int d = iterNum - 1; d >= 0; d--) {
                kernelEfficientScanDownSweep<<<blockNum, blockSize>>>(tmpLength, d, dev_indices);
                checkCUDAError("kernelEfficientScanDownSweep failed!");
            }

            //scatter
            Common::kernScatter<<<originalBlockNum, blockSize>>>(n, dev_odata, dev_idata, dev_bools, dev_indices);
            checkCUDAError("kernScatter failed!");

            timer().endGpuTimer();

            // get total non-zero count
            int count;
            cudaMemcpy(&count, dev_indices + n - 1, sizeof(int), cudaMemcpyDeviceToHost);
            count += (int)(idata[n - 1] != 0);

            // copy array from gpu to cpu
            cudaMemcpy(odata, dev_odata, count * sizeof(int), cudaMemcpyDeviceToHost);
            checkCUDAError("cudaMemcpy final odata failed!");

            // free memory
            cudaFree(dev_idata);
            cudaFree(dev_odata);
            cudaFree(dev_bools);
            cudaFree(dev_indices);

            return count;
        }
    }
}

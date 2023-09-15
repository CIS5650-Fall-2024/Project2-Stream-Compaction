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

        __global__ void kernalEfficientScan_UpSweep(int k, int step, int* data)
        {
            int index = threadIdx.x + (blockDim.x * blockIdx.x);

            if (index >= k) return;
            data[(index + 1) * step - 1] = data[(index + 1) * step - 1] + data[(index + 1) * step - 1 - (step >> 1)];
        }
        __global__ void kernalEfficientScan_DownSweep(int k, int step, int* data)
        {
            int index = threadIdx.x + (blockDim.x * blockIdx.x);

            if (index >= k) return;
            int temp = data[(index + 1) * step - 1];
            data[(index + 1) * step - 1] = data[(index + 1) * step - 1] + data[(index + 1) * step - 1 - (step >> 1)];
            data[(index + 1) * step - 1 - (step >> 1)] = temp;
        }

        __global__ void kernalLabelData(int n, int* label, const int* data)
        {
            int index = threadIdx.x + (blockDim.x * blockIdx.x);
            if (index >= n) return;

            label[index] = (data[index] != 0 ? 1 : 0);
        }

        __global__ void kernalScatter(int n, int* result, const int* label, const int* data)
        {
            int index = threadIdx.x + (blockDim.x * blockIdx.x);
            if (index >= n) return;

            if (data[index] != 0)
            {
                result[label[index]] = data[index];
            }
        }

        void EfficientParallelScan(const int& pot_length, int* dev_data)
        {
            // Up-Sweep
            int k = pot_length >> 1;
            int step = 2;
            while (k > 1)
            {
                kernalEfficientScan_UpSweep << <(k + 31) / 32, 32 >> > (k, step, dev_data);
                checkCUDAError("Luanch kernalEfficientScan_UpSweep failed!");

                k >>= 1;
                step <<= 1;
            }
            //replace last number with 0
            cudaMemset(dev_data + pot_length - 1, 0, sizeof(int));

            // Down-Sweep
            k = 1;
            step = pot_length;
            while (k < pot_length)
            {
                kernalEfficientScan_DownSweep << <(k + 31) / 32, 32 >> > (k, step, dev_data);
                checkCUDAError("Luanch kernalEfficientScan_DownSweep failed!");

                k <<= 1;
                step >>= 1;
            }
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            int pot_length = pow(2, ilog2ceil(n));// power-of-two length;

            // Malloc necessary space on GPU
            int* dev_data;

            cudaMalloc((void**)&dev_data, pot_length * sizeof(int));
            checkCUDAError("cudaMalloc dev_data failed!");

            // copy data from CPU to GPU
            cudaMemcpy(dev_data, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            checkCUDAError("Memcpy idata(host) to dev_data(device) failed!");

            timer().startGpuTimer();
            EfficientParallelScan(pot_length, dev_data);
            timer().endGpuTimer();

            // copy data back to CPU
            cudaMemcpy(odata, dev_data, n * sizeof(int), cudaMemcpyDeviceToHost);
            checkCUDAError("Memcpy dev_data(device) to odata(host) failed!");

            // free memory on GPU
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
            int pot_length = pow(2, ilog2ceil(n));// power-of-two length;

            // Malloc necessary space on GPU
            int* dev_data;
            int* dev_label;
            int* dev_result;

            cudaMalloc((void**)&dev_data, pot_length * sizeof(int));
            checkCUDAError("cudaMalloc dev_data failed!");

            cudaMalloc((void**)&dev_label, pot_length * sizeof(int));
            checkCUDAError("cudaMalloc dev_label failed!");

            cudaMalloc((void**)&dev_result, n * sizeof(int));
            checkCUDAError("cudaMalloc dev_result failed!");

            // copy data from CPU to GPU
            cudaMemcpy(dev_data, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            checkCUDAError("Memcpy idata(host) to dev_data(device) failed!");

            timer().startGpuTimer();

            kernalLabelData << <(pot_length + 31) / 32, 32 >> > (pot_length, dev_label, dev_data);

            EfficientParallelScan(pot_length, dev_label);

            kernalScatter << <(pot_length + 31) / 32, 32 >> > (pot_length, dev_result, dev_label, dev_data);
            checkCUDAError("Luanch kernalScatter failed!");

            timer().endGpuTimer();

            int num_remain;
            cudaMemcpy(&num_remain, dev_label + pot_length - 1, sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(odata, dev_result, num_remain * sizeof(int), cudaMemcpyDeviceToHost);

            cudaFree(dev_data);
            cudaFree(dev_result);
            cudaFree(dev_label);
            
            return num_remain;
        }
    }
}

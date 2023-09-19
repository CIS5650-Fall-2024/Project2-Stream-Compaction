#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/scan.h>
#include "common.h"
#include "radix.h"

namespace StreamCompaction {
    namespace Radix {
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

        void EfficientParallelScan(const int& pot_length, int* dev_data)
        {
            // Up-Sweep
            int k = pot_length >> 1;
            int step = 2;
            while (k > 1)
            {
                const int block = BlockSize;
                kernalEfficientScan_UpSweep << <(k + block - 1) / block, block >> > (k, step, dev_data);
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
                const int block = BlockSize;
                kernalEfficientScan_DownSweep << <(k + block - 1) / block, block >> > (k, step, dev_data);
                checkCUDAError("Luanch kernalEfficientScan_DownSweep failed!");

                k <<= 1;
                step >>= 1;
            }
        }

        __global__ void kernalCheckStop(int n, const int* idata, int* stop)
        {
            int index = threadIdx.x + (blockDim.x * blockIdx.x);
            if (index >= n - 1) return;

            if (idata[index] > idata[index + 1]) (*stop) = 1;
        }

        __global__ void kernalRadixMapToBoolean(int n, int k, int* label, const int* idata, int* skip) {
            int index = threadIdx.x + (blockDim.x * blockIdx.x);
            if (index >= n) return;

            int num = idata[index];
            int result = ((num & (1 << k)) != 0 ? 0 : 1);

            if (k == 0 || result != ((num & (1 << (k - 1))) != 0 ? 0 : 1))
            {
                *skip = 1;
            }
            label[index] = result;
        }
        __global__ void kernalRadixScattering(int n, int k, int start, int* odata, const int* idata, const int* label)
        {
            int index = threadIdx.x + (blockDim.x * blockIdx.x);
            if (index >= n) return;

            bool result = ((idata[index] & (1 << k)) != 0 ? 1 : 0);
            if (result)
            {
                odata[start + index - label[index]] = idata[index];
            }
            else
            {
                odata[label[index]] = idata[index];
            }
        }
        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void sort(int n, int *odata, const int *idata) 
        {
            int pot_length = pow(2, ilog2ceil(n));// power-of-two length;

            int* dev_read;
            int* dev_write;
            int* dev_label;
            int* dev_number;

            cudaMalloc((void**)&dev_read, pot_length * sizeof(int));
            checkCUDAError("cudaMalloc dev_read failed!");
            cudaMalloc((void**)&dev_write, pot_length * sizeof(int));
            checkCUDAError("cudaMalloc dev_write failed!");
            cudaMalloc((void**)&dev_label, pot_length * sizeof(int));
            checkCUDAError("cudaMalloc dev_label failed!");
            cudaMalloc((void**)&dev_number, sizeof(int));
            checkCUDAError("cudaMalloc dev_number failed!");

            cudaMemset(dev_read, (1 << 8) - 1, pot_length * sizeof(int));
            cudaMemcpy(dev_read, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            checkCUDAError("Memcpy idata(host) to dev_read(device) failed!");
            timer().startGpuTimer();
            for (int i = 0; i < 32; ++i)
            {
                // check whether to stop
                cudaMemset(dev_number, 0, sizeof(int));
                kernalCheckStop << < (pot_length + BlockSize - 1) / BlockSize, BlockSize >> > (pot_length, dev_read, dev_number);
                int stop;
                cudaMemcpy(&stop, dev_number, sizeof(int), cudaMemcpyDeviceToHost);
                if (stop == 0) break;

                // label and check whether to skip this bit
                cudaMemset(dev_number, 0, sizeof(int));
                kernalRadixMapToBoolean << < (pot_length + BlockSize - 1) / BlockSize, BlockSize >> > (pot_length, i, dev_label, dev_read, dev_number);
                checkCUDAError("Luanch kernalRadixMapToBoolean failed!");

                int skip;
                cudaMemcpy(&skip, dev_number, sizeof(int), cudaMemcpyDeviceToHost);
                if (skip == 0) continue;

                // read the last number of label_1 back
                int last_num;
                cudaMemcpy(&last_num, dev_label + pot_length - 1, sizeof(int), cudaMemcpyDeviceToHost);

                EfficientParallelScan(pot_length, dev_label);

                int start_index;
                cudaMemcpy(&start_index, dev_label + pot_length - 1, sizeof(int), cudaMemcpyDeviceToHost);
                start_index += last_num;

                kernalRadixScattering << < (pot_length + BlockSize - 1) / BlockSize, BlockSize >> > (pot_length, i, start_index, dev_write, dev_read, dev_label);
                checkCUDAError("Luanch kernalRadixMapToBoolean failed!");

                std::swap(dev_write, dev_read);
            }
            timer().endGpuTimer();
            cudaMemcpy(odata, dev_read, n * sizeof(int), cudaMemcpyDeviceToHost);

            cudaFree(dev_read);
            cudaFree(dev_write);
            cudaFree(dev_label);
            cudaFree(dev_number);
        }
    }
}

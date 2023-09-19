#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

#define block_size 512

namespace StreamCompaction {
    namespace Efficient {
        using StreamCompaction::Common::PerformanceTimer;
        using StreamCompaction::Common::kernMapToBoolean;
        using StreamCompaction::Common::kernScatter;

        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        __global__ void kern_scan_up_sweep(int d, int n, int *data)
        {
            /* Memory access errors occur when using int or long on large array sizes
               This is most likely due to overflow when converting threadIdx to 1D
               And further multiplying by 2^d, this is done to iterate by 2^(d+1) */
			unsigned long long index = threadIdx.x + (blockIdx.x * blockDim.x);
            index <<= (d + 1);
            if (index >= n)
            {
                return; 
            }
            data[index + (1 << (d + 1)) - 1] += data[index + (1 << d) - 1];
        }

        __global__ void kern_scan_down_sweep(int d, int n, int* data)
        {
            unsigned long long index = threadIdx.x + (blockIdx.x * blockDim.x);
            index <<= (d + 1);
            if (index >= n)
            {
                return;
            }
            int temp = data[index + (1 << d) - 1];
            data[index + (1 << d) - 1] = data[index + (1 << (d + 1)) - 1];
            data[index + (1 << (d + 1)) - 1] += temp;
        }

        __global__ void kern_add_padding_zeros(int n, int full_data_size, int* data)
        {
            int index = threadIdx.x + (blockIdx.x * blockDim.x);
            if (index >= full_data_size || index < n)
            {
				return;
			}
            data[index] = 0;
		}


        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {

            int* dev_odata;

            int n_ceil_log_2 = ilog2ceil(n);
            int full_data_size = (1 << n_ceil_log_2);

            dim3 full_blocks_per_grid((full_data_size + block_size - 1) / block_size);

            cudaMalloc((void**)&dev_odata, full_data_size * sizeof(int));
            checkCUDAErrorFn("cudaMalloc dev_odata failed!");

            cudaMemcpy(dev_odata, idata, sizeof(int) * n, cudaMemcpyHostToDevice);
            checkCUDAErrorFn("cudaMemcpy dev_odata failed!");

            timer().startGpuTimer();
            kern_add_padding_zeros <<<full_blocks_per_grid, block_size>>> (n, full_data_size, dev_odata);
            for (int i = 0; i < n_ceil_log_2; i++)
            {
                kern_scan_up_sweep <<<full_blocks_per_grid, block_size>>> (i, full_data_size, dev_odata);
            }
            cudaMemset(&dev_odata[full_data_size - 1], 0, sizeof(int));
            checkCUDAErrorFn("cudaMemset odata last ele failed!");

            for (int i = n_ceil_log_2 - 1; i >= 0; i--)
            {
				kern_scan_down_sweep <<<full_blocks_per_grid, block_size>>> (i, full_data_size, dev_odata);
			}
            timer().endGpuTimer();

            cudaMemcpy(odata, dev_odata, sizeof(int) * n, cudaMemcpyDeviceToHost);
            checkCUDAErrorFn("cudaMemcpy odata failed!");

            cudaFree(dev_odata);
            checkCUDAErrorFn("cudaFree dev_odata failed!");
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
            
            int *dev_idata;
            int *dev_odata;
            int *dev_scan_result;
            int *dev_temp_bools;

            int n_ceil_log_2 = ilog2ceil(n);
            int full_data_size = (1 << n_ceil_log_2);

            dim3 n_blocks_per_grid((n + block_size - 1) / block_size);
            dim3 full_blocks_per_grid((full_data_size + block_size - 1) / block_size);

            cudaMalloc((void**)&dev_odata, n * sizeof(int));
            checkCUDAErrorFn("cudaMalloc dev_odata failed!");

            cudaMalloc((void**)&dev_idata, n * sizeof(int));
			checkCUDAErrorFn("cudaMalloc dev_idata failed!");

            cudaMalloc((void**)&dev_scan_result, full_data_size * sizeof(int));
            checkCUDAErrorFn("cudaMalloc dev_scan_result failed!");

            cudaMalloc((void**)&dev_temp_bools, n * sizeof(int));
            checkCUDAErrorFn("cudaMalloc dev_temp_bools failed!");

            cudaMemcpy(dev_idata, idata, sizeof(int) * n, cudaMemcpyHostToDevice);
            checkCUDAErrorFn("cudaMemcpy dev_odata failed!");

            timer().startGpuTimer();
            // map to boolean
            kernMapToBoolean<<<n_blocks_per_grid, block_size>>>(n, dev_temp_bools, dev_idata);
            cudaMemcpy(dev_scan_result, dev_temp_bools, sizeof(int) * n, cudaMemcpyDeviceToDevice);

            // perform scan
            kern_add_padding_zeros << <full_blocks_per_grid, block_size >> > (n, full_data_size, dev_scan_result);
            for (int i = 0; i < n_ceil_log_2; i++)
            {
                kern_scan_up_sweep << <full_blocks_per_grid, block_size >> > (i, full_data_size, dev_scan_result);
            }
            cudaMemset(&dev_scan_result[full_data_size - 1], 0, sizeof(int));
            for (int i = n_ceil_log_2 - 1; i >= 0; i--)
            {
                kern_scan_down_sweep << <full_blocks_per_grid, block_size >> > (i, full_data_size, dev_scan_result);
            }
            int odata_size;
            cudaMemcpy(&odata_size, &dev_scan_result[full_data_size - 1], sizeof(int), cudaMemcpyDeviceToHost);

            // scatter
            kernScatter<<<n_blocks_per_grid, block_size>>>(n, dev_odata, dev_idata, dev_temp_bools, dev_scan_result);
            timer().endGpuTimer();

            cudaMemcpy(odata, dev_odata, sizeof(int) * odata_size, cudaMemcpyDeviceToHost);

            cudaFree(dev_idata);
            checkCUDAErrorFn("cudaFree dev_idata failed!");

            cudaFree(dev_odata);
            checkCUDAErrorFn("cudaFree dev_odata failed!");

            cudaFree(dev_scan_result);
            checkCUDAErrorFn("cudaFree dev_scan_result failed!");

            cudaFree(dev_temp_bools);
            checkCUDAErrorFn("cudaFree dev_temp_bools failed!");

            return odata_size;
        }
    }
}

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

        __global__ void kernScanUpsweep(int n, int d, int* idata) {
            int k = threadIdx.x + (blockIdx.x * blockDim.x);
            if (k >= n || k % (1 << (d + 1)) != 0) {
                return;
            }
            int idx_right = k + (1 << (d + 1)) - 1;
            int idx_left = k + (1 << d) - 1;
            idata[idx_right] += idata[idx_left];
        }

        __global__ void kernScanDownsweep(int n, int d, int* idata) {
            int k = threadIdx.x + (blockIdx.x * blockDim.x);
            if (k >= n || k % (1 << (d + 1)) != 0) {
                return;
            }
            int idx_right = k + (1 << (d + 1)) - 1;
            int idx_left = k + (1 << d) - 1;

            int tmp = idata[idx_left];
            idata[idx_left] = idata[idx_right];
            idata[idx_right] += tmp;
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            

            int n_padded = 1 << ilog2ceil(n);

            int *dev_idata;

            cudaMalloc((void**)&dev_idata, n_padded * sizeof(int));
            checkCUDAError("cudaMalloc dev_idata failed!");
            
            cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            checkCUDAError("cudaMemcpy dev_idata failed!");
            cudaMemset(dev_idata + n, 0, (n_padded - n) * sizeof(int));
            timer().startGpuTimer();
            _scan_dev(n_padded, dev_idata);
            timer().endGpuTimer();

            cudaMemcpy(odata, dev_idata+1, (n - 1) * sizeof(int), cudaMemcpyDeviceToHost);
            odata[n - 1] = odata[n - 2] + idata[n - 1];
            checkCUDAError("cudaMemcpy odata failed!");
            cudaFree(dev_idata);
            checkCUDAError("cudaFree failed!");     
        }

        void _scan_dev(int n_padded, int *dev_idata) {
            int blockSize = 128;
            int num_blocks = (n_padded + blockSize - 1) / blockSize;

            int ilog2ceil_n = ilog2ceil(n_padded)-1;

            for (int d = 0; d <= ilog2ceil_n; d++) {
                kernScanUpsweep<<<num_blocks, blockSize>>>(n_padded, d, dev_idata);
                checkCUDAError("kernScanUpsweep failed!");
            }
            // 2. downsweep
            cudaMemset(dev_idata + n_padded - 1, 0, sizeof(int));
            checkCUDAError("cudaMemset dev_idata failed!");

            for (int d = ilog2ceil_n; d >= 0; d--) {
                kernScanDownsweep<<<num_blocks, blockSize>>>(n_padded, d, dev_idata);
                checkCUDAError("kernScanDownsweep failed!");
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
            int blockSize = 128;
            int n_padded = 1 << ilog2ceil(n);
            int num_blocks = (n_padded + blockSize - 1) / blockSize;

            int *dev_bools; 
            int *dev_idata;           
            int *dev_indices;
            int *dev_scattered;

            cudaMalloc((void**)&dev_bools, n_padded * sizeof(int));
            checkCUDAError("cudaMalloc dev_bools failed!");
            cudaMalloc((void**)&dev_idata, n_padded * sizeof(int));
            checkCUDAError("cudaMalloc dev_idata failed!");
            cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            checkCUDAError("cudaMemcpy dev_idata failed!");
            if (n < n_padded) {
                cudaMemset(dev_idata + n, 0, (n_padded - n) * sizeof(int));
                checkCUDAError("cudaMemset dev_idata failed!");
            }

            timer().startGpuTimer();
            Common::kernMapToBoolean<<<num_blocks, blockSize>>>(n_padded, dev_bools, dev_idata);

            cudaMalloc((void**)&dev_indices, n_padded * sizeof(int));
            checkCUDAError("cudaMalloc dev_indices failed!");
            cudaMemcpy(dev_indices, dev_bools, n_padded * sizeof(int), cudaMemcpyDeviceToDevice);
            checkCUDAError("cudaMemcpy dev_indices failed!");

            _scan_dev(n_padded, dev_indices);
            checkCUDAError("_scan_dev failed!");

            cudaMalloc((void**)&dev_scattered, n_padded * sizeof(int));
            checkCUDAError("cudaMalloc dev_scattered failed!");

            Common::kernScatter<<<num_blocks, blockSize>>>(n_padded, dev_scattered, dev_idata, dev_bools, dev_indices);
            checkCUDAError("kernScatter failed!");
            timer().endGpuTimer();

            // since dev_indices is exclusive scan (prefix sum), we can grab num valid elements from the last element
            int num_valid;
            int last_bool;
            cudaMemcpy(&num_valid, dev_indices + n_padded - 1, sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(&last_bool, dev_bools + n_padded - 1, sizeof(int), cudaMemcpyDeviceToHost);
            checkCUDAError("cudaMemcpy num_valid failed!");
            num_valid += last_bool;
            cudaMemcpy(odata, dev_scattered, num_valid * sizeof(int), cudaMemcpyDeviceToHost);
            checkCUDAError("cudaMemcpy odata failed!");

            cudaFree(dev_bools);
            cudaFree(dev_idata);
            cudaFree(dev_indices);
            cudaFree(dev_scattered);
            checkCUDAError("cudaFree failed!");
            
            return num_valid;
        }
    }
}

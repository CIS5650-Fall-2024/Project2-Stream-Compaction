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
            if (k >= n || k % (2 << (d + 1)) != 0) {
                return;
            }
            int idx_right = k + (2 << (d + 1)) - 1;
            int idx_left = k + (2 << d) - 1;
            idata[idx_right] += idata[idx_left];
        }

        __global__ void kernScanDownsweep(int n, int d, int* idata) {
            int k = threadIdx.x + (blockIdx.x * blockDim.x);
            if (k >= n || k % (2 << (d + 1)) != 0) {
                return;
            }
            int idx_right = k + (2 << (d + 1)) - 1;
            int idx_left = k + (2 << d) - 1;
            
            int tmp = idata[idx_left];
            idata[idx_left] = idata[idx_right];
            idata[idx_right] += tmp;
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            timer().startGpuTimer();

            int blockSize = 128;
            int n_padded = 1 << ilog2ceil(n);
            int num_blocks = (n_padded + blockSize - 1) / blockSize;

            int *dev_idata;

            cudaMalloc((void**)&dev_idata, n_padded * sizeof(int));
            checkCUDAError("cudaMalloc dev_idata failed!");
            
            cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            checkCUDAError("cudaMemcpy dev_idata failed!");

            // 1. upsweep
            for (int d = 0; d <= ilog2ceil(n_padded) - 1; d++) {
                kernScanUpsweep<<<num_blocks, blockSize>>>(n_padded, d, dev_idata);
            }
            // 2. downsweep
            cudaMemset(dev_idata + n_padded - 1, 0, sizeof(int));

            for (int d = ilog2ceil(n_padded) - 1; d >= 0; d--) {
                kernScanDownsweep<<<num_blocks, blockSize>>>(n_padded, d, dev_idata);
            }

            cudaMemcpy(odata, dev_idata, n * sizeof(int), cudaMemcpyDeviceToHost);

            timer().endGpuTimer();
        }

        void _scan_dev(int n, int *dev_idata) {
            int blockSize = 128;
            int n_padded = 1 << ilog2ceil(n);
            int num_blocks = (n_padded + blockSize - 1) / blockSize;

            for (int d = 0; d <= ilog2ceil(n_padded) - 1; d++) {
                kernScanUpsweep<<<num_blocks, blockSize>>>(n_padded, d, dev_idata);
            }
            // 2. downsweep
            cudaMemset(dev_idata + n_padded - 1, 0, sizeof(int));

            for (int d = ilog2ceil(n_padded) - 1; d >= 0; d--) {
                kernScanDownsweep<<<num_blocks, blockSize>>>(n_padded, d, dev_idata);
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
            timer().startGpuTimer();
            int blockSize = 128;
            int n_padded = 1 << ilog2ceil(n);
            int num_blocks = (n_padded + blockSize - 1) / blockSize;

            int *dev_bools;            
            int *dev_indices;
            int *dev_scattered;

            cudaMalloc((void**)&dev_bools, n_padded * sizeof(int));
            checkCUDAError("cudaMalloc dev_bools failed!");

            kernMapToBoolean<<<num_blocks, blockSize>>>(n_padded, dev_bools, dev_idata);

            cudaMalloc((void**)&dev_indices, n_padded * sizeof(int));
            checkCUDAError("cudaMalloc dev_indices failed!");
            cudaMemcpy(dev_indices, dev_bools, n_padded * sizeof(int), cudaMemcpyDeviceToDevice);

            _scan_dev(n_padded, dev_indices);

            cudaMalloc((void**)&dev_scattered, n_padded * sizeof(int));

            kernScatter<<<num_blocks, blockSize>>>(n_padded, dev_scattered, dev_idata, dev_bools, dev_indices);

            // since dev_indices is exclusive scan (prefix sum), we can grab num valid elements from the last element
            int num_valid;
            int last_bool;
            cudaMemcpy(&num_valid, dev_indices + n_padded - 1, sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(&last_bool, dev_bools + n_padded - 1, sizeof(int), cudaMemcpyDeviceToHost);
            num_valid += last_bool;
            cudaMemcpy(odata, dev_idata, num_valid * sizeof(int), cudaMemcpyDeviceToHost);

            cudaFree(dev_bools);
            cudaFree(dev_indices);
            cudaFree(dev_scattered);
            timer().endGpuTimer();
            return num_valid;
        }
    }
}

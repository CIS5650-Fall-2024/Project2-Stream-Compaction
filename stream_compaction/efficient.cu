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

        __global__ void upSweepKernel(int n, int d, int *data){
            int index = threadIdx.x + (blockIdx.x * blockDim.x);
            if (index >= n){
                return;
            }
            if ((index + 1) % (1 << (d + 1)) == 0){
                data[index] += data[index - (1 << d)];
            }
        }

        __global__ void downSweepKernel(int n, int d, int *data){
            int index = threadIdx.x + (blockIdx.x * blockDim.x);
            if (index >= n){
                return;
            }
            if ((index + 1) % (1 << (d + 1)) == 0){
                int root = data[index];
                int left_index = index - (1 << d);
                data[index] += data[left_index];
                data[left_index] = root;
            }

        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            int d_round = ilog2ceil(n);
            int full_size = 1 << d_round;
            int block_size = 512;
            dim3 fullBlocksPerGrid((block_size + full_size - 1) / block_size);

            int *d_data;
            cudaMalloc((void **)&d_data, full_size * sizeof(int));
            cudaMemset(d_data, 0, full_size * sizeof(int));
            cudaMemcpy(d_data, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            // Up-sweep
            timer().startGpuTimer();
            for (int d = 0; d < d_round; d++){
                upSweepKernel<<<fullBlocksPerGrid, block_size>>>(full_size, d, d_data);
            }
            // Down-sweep
            cudaMemset(d_data + full_size - 1, 0, sizeof(int));
            for (int d = d_round - 1; d >= 0; d--){
                downSweepKernel<<<fullBlocksPerGrid, block_size>>>(full_size, d, d_data);
            }
            timer().endGpuTimer();
            // Copy result to odata
            cudaMemcpy(odata, d_data, n * sizeof(int), cudaMemcpyDeviceToHost);
            cudaFree(d_data);
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
            int *scatter_result = new int[n];
            int *d_idata, *d_bools, *d_odata;
            cudaMalloc((void **)&d_idata, n * sizeof(int));
            cudaMalloc((void **)&d_bools, n * sizeof(int));
            cudaMalloc((void **)&d_odata, n * sizeof(int));
            cudaMemcpy(d_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            int block_size = 512;
            dim3 fullBlocksPerGrid((block_size + n - 1) / block_size);
            // efficient scan
            int d_round = ilog2ceil(n);
            int full_size = 1 << d_round;
            dim3 scanBlocksPerGrid((block_size + full_size - 1) / block_size);
            int *d_scan_buffer;
            cudaMalloc((void **)&d_scan_buffer, full_size * sizeof(int));
            cudaMemset(d_scan_buffer, 0, full_size * sizeof(int));

            timer().startGpuTimer();
            Common::kernMapToBoolean<<<fullBlocksPerGrid, block_size>>>(n, d_bools, d_idata);
            cudaMemcpy(d_scan_buffer, d_bools, n * sizeof(int), cudaMemcpyDeviceToDevice);
            for (int d = 0; d < d_round; d++){
                upSweepKernel<<<scanBlocksPerGrid, block_size>>>(full_size, d, d_scan_buffer);
            }
            cudaMemset(d_scan_buffer + full_size - 1, 0, sizeof(int));
            for (int d = d_round - 1; d >= 0; d--){
                downSweepKernel<<<scanBlocksPerGrid, block_size>>>(full_size, d, d_scan_buffer);
            }
            // scatter
            Common::kernScatter<<<fullBlocksPerGrid, block_size>>>(n, d_odata, d_idata, d_bools, d_scan_buffer);
            timer().endGpuTimer();
            // copy result
            cudaMemcpy(odata, d_odata, n * sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(scatter_result, d_scan_buffer, n * sizeof(int), cudaMemcpyDeviceToHost);
            cudaFree(d_idata);
            cudaFree(d_bools);
            cudaFree(d_odata);
            cudaFree(d_scan_buffer);

            return scatter_result[n - 1] + (idata[n - 1] != 0);
        }
    }
}

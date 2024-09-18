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

        __global__ void kernUpsweep(int n, int d, int* data) {
            int k = (blockDim.x * blockIdx.x) + threadIdx.x;
            int pow_d_plus_one = (1 << d + 1);

            if (k >= n / pow_d_plus_one) {
                return;
            }
            
            k *= pow_d_plus_one;

            int pow_d = (1 << d);

            data[k + pow_d_plus_one - 1] += data[k + pow_d - 1];
        }

        __global__ void kernDownsweep(int n, int d, int* data) {
            int k = (blockDim.x * blockIdx.x) + threadIdx.x;
            int pow_d_plus_one = (1 << d + 1);

            if (k >= n / pow_d_plus_one) {
                return;
            }

            k *= pow_d_plus_one;

            int curr_left_idx = k + (1 << d) - 1;
            int left_child_val = data[curr_left_idx];       // Save left child
            int curr_node_idx = k + pow_d_plus_one - 1;
            data[curr_left_idx] = data[curr_node_idx];      // Set left child to this node’s value
            data[curr_node_idx] += left_child_val;          // Set right child to old left value + this node’s value
        }

        __global__ void kernPadZeroes(int original_length, int start_point, int* data) {
            int k = (blockDim.x * blockIdx.x) + threadIdx.x;
            if (k >= original_length && k < start_point) {
                data[k] = 0;
            }
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            int log = ilog2ceil(n);
            int power_two_len = (1 << log);

            int* data;
            cudaMalloc((void**)&data, sizeof(int) * power_two_len);
            cudaMemcpy(data, idata, sizeof(int) * power_two_len, cudaMemcpyHostToDevice);

            timer().startGpuTimer();

            int blockSize = 128;
            dim3 blockDim = dim3((power_two_len + blockSize - 1) / blockSize);
            kernPadZeroes << <blockDim, blockSize >> > (n, power_two_len, data);

            int threads_to_launch;

            //upsweep
            for (int d = 0; d < log; d++) {
                threads_to_launch = power_two_len / (1 << d + 1);

                blockDim = dim3((threads_to_launch + blockSize - 1) / blockSize);

                kernUpsweep << <blockDim, blockSize >> > (power_two_len, d, data);
            }

            //set root to 0
            cudaMemset(data + power_two_len - 1, 0, sizeof(int));

            //downsweep
            for (int d = log - 1; d >= 0; d--) {
                threads_to_launch = power_two_len / (1 << d + 1);

                blockDim = dim3((threads_to_launch + blockSize - 1) / blockSize);

                kernDownsweep << <blockDim, blockSize >> > (power_two_len, d, data);
            }

            timer().endGpuTimer();

            cudaMemcpy(odata, data, sizeof(int) * power_two_len, cudaMemcpyDeviceToHost);
            cudaFree(data);
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

            int num_elts = -1;

            int log = ilog2ceil(n);
            int power_two_len = (1 << log);

            int blockSize = 128;
            dim3 blockDim((n + blockSize - 1) / blockSize);

            int* in_data;
            cudaMalloc((void**)&in_data, sizeof(int) * n);
            cudaMemcpy(in_data, idata, sizeof(int) * n, cudaMemcpyHostToDevice);

            int* bools;
            cudaMalloc((void**)&bools, sizeof(int) * power_two_len);

            int* out_data;
            cudaMalloc((void**)&out_data, sizeof(int) * n);

            timer().startGpuTimer();

            int scan_blockSize = 128;
            dim3 scan_blockDim = dim3((power_two_len + blockSize - 1) / blockSize);

            kernPadZeroes << <scan_blockDim, scan_blockSize >> > (n, power_two_len, bools);

            StreamCompaction::Common::kernMapToBoolean << < blockDim, blockSize >> >(power_two_len, bools, in_data);

            //START SCAN (input data is now bools)

            int* scan_data;
            cudaMalloc((void**)&scan_data, sizeof(int) * power_two_len);
            cudaMemcpy(scan_data, bools, sizeof(int) * power_two_len, cudaMemcpyDeviceToDevice);

            int threads_to_launch;

            //upsweep
            for (int d = 0; d < log; d++) {
                threads_to_launch = power_two_len / (1 << d + 1);

                scan_blockDim = dim3((threads_to_launch + scan_blockSize - 1) / scan_blockSize);

                kernUpsweep << <scan_blockDim, scan_blockSize >> > (power_two_len, d, scan_data);
            }

            //set root to 0
            cudaMemset(scan_data + power_two_len - 1, 0, sizeof(int));

            //downsweep
            for (int d = log - 1; d >= 0; d--) {
                threads_to_launch = power_two_len / (1 << d + 1);

                scan_blockDim = dim3((threads_to_launch + scan_blockSize - 1) / scan_blockSize);

                kernDownsweep << <scan_blockDim, scan_blockSize >> > (power_two_len, d, scan_data);
            }

            //END SCAN -- scan is in place so scan_data has the output

            StreamCompaction::Common::kernScatter << < blockDim, blockSize >> >(power_two_len, out_data, in_data, bools, scan_data);

            timer().endGpuTimer();

            cudaMemcpy(odata, out_data, sizeof(int) * n, cudaMemcpyDeviceToHost);
            cudaMemcpy(&num_elts, scan_data + power_two_len - 1, sizeof(int), cudaMemcpyDeviceToHost);

            int last_bool;
            cudaMemcpy(&last_bool, bools + power_two_len - 1, sizeof(int), cudaMemcpyDeviceToHost);
            if (last_bool == 1) num_elts++;

            cudaFree(bools);
            cudaFree(in_data);
            cudaFree(out_data);
            cudaFree(scan_data);

            return num_elts;
        }
    }
}

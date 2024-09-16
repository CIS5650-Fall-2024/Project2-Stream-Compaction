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
            if (k > n) {
                return;
            }
            /*int pow_res = pow(2, d - 1);
            if (k >= pow_res) {
                int idx = k + pow(2, d + 1) - 1;
                data[idx] += data[k + d];
            }*/
            int pow_d_plus_one = powf(2, d + 1);
            int pow_d = powf(2, d);
            data[k + pow_d_plus_one - 1] += data[k + pow_d - 1];
        }

        __global__ void kernDownsweep(int n, int d, int* data) {
            int k = (blockDim.x * blockIdx.x) + threadIdx.x;
            if (k > n) {
                return;
            }
            int curr_left_idx = k + pow(2, d) - 1;
            int left_child = data[curr_left_idx];       // Save left child
            int curr_node_idx = k + pow(2, d + 1) - 1;
            data[curr_left_idx] = data[curr_node_idx];  // Set left child to this node’s value
            data[curr_node_idx] += left_child;          // Set right child to old left value + this node’s value
        }

        __global__ void kernPadZeroes(int n, int d, int* data) {
            int k = (blockDim.x * blockIdx.x) + threadIdx.x;
            if (k > n) {
                return;
            }
            if (k > d) {
                data[k] = 0;
            }
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            timer().startGpuTimer();
            // TODO

            int original_length = n;
            int log = ilog2ceil(n);
            float f_log = log2f(n);

            //size of array is not power of 2
            if (f_log != (float)log) {
                int new_length = pow(2, log);
                n = new_length;
            }

            int* data;
            cudaMalloc((void**)&data, sizeof(int) * n);
            cudaMemcpy(data, idata, sizeof(int) * n, cudaMemcpyHostToDevice);

            int blockSize = 128;
            dim3 blockDim((n + blockSize - 1) / blockSize);

            if (f_log != (float)log) {
                kernPadZeroes<<<blockDim, blockSize>>>(n, original_length, data);
            }

            for (int d = 1; d <= log; d++) {
                kernUpsweep << <blockDim, blockSize >> > (n, d, data);
            }

            //set root to 0
            cudaMemset(data + n - 1, 0, sizeof(int));

            for (int d = log - 1; d >= 0; d--) {
                kernDownsweep << <blockDim, blockSize >> > (n, d, data);
            }

            cudaMemcpy(odata, data, sizeof(int) * n, cudaMemcpyDeviceToHost);
            cudaFree(data);

            timer().endGpuTimer();
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

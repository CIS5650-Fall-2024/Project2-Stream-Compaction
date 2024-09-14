#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"

#include <thrust/device_vector.h>

namespace StreamCompaction {
    namespace Naive {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        __global__ void naive_scan_block(int n, int *data, int *block_sums) {
            extern __shared__ int shared[];

            const auto index = blockIdx.x * blockDim.x + threadIdx.x;
            const auto shared_index = threadIdx.x * 2;
            if (index >= n) return;
            
            // load data interleaved into shared memory
            shared[shared_index] = data[index];
            __syncthreads();

            // NOTE(rahul): I make the input and output offsets the reverse of what
            // they usually are so that when we execute the first for loop, we are correct
            // and when we exit, we can use output offset, which is semantically
            // correct.
            auto input_offset = 1;
            auto output_offset = 0;

            for (int d = 1; d < blockDim.x << 1; d <<= 1) {

                input_offset = 1 - input_offset;
                output_offset = 1 - output_offset;

                if (threadIdx.x >= d) {
                    shared[shared_index + output_offset] = shared[shared_index - d * 2 + input_offset] + shared[shared_index + input_offset]; 
                } else {
                    shared[shared_index + output_offset] = shared[shared_index + input_offset];
                }
                __syncthreads();
            }

            if (threadIdx.x == 0) {
                data[blockIdx.x * blockDim.x] = 0;
                if (block_sums) block_sums[blockIdx.x] = shared[(blockDim.x - 1) * 2 + output_offset];
            } else {
                data[index] = shared[(threadIdx.x - 1) * 2 + output_offset];
            }

        }

        void _scan(int n, const thrust::device_ptr<int> &data, int block_size = 128) {
            const auto num_blocks = (n + block_size - 1) / block_size;

            thrust::device_vector<int> block_sums(num_blocks);

            naive_scan_block<<<num_blocks, block_size, 2 * block_size * sizeof(int)>>>(
                n, thrust::raw_pointer_cast(data),
                thrust::raw_pointer_cast(block_sums.data())
            );

            if (num_blocks == 1) {
                return;
            } else if (num_blocks <= block_size) {
                const auto block_sum_num_blocks = (num_blocks + block_size - 1) / block_size;
                naive_scan_block<<<block_sum_num_blocks, block_size, 2 * block_size * sizeof(int)>>>(
                    num_blocks,
                    thrust::raw_pointer_cast(block_sums.data())
                );
            } else {
                _scan(num_blocks, block_sums.data(), block_size);
            }

            const auto final_scan_num_blocks = (n - 1) / block_size;
            Common::resolve_scan_blocks<<<num_blocks, block_size, 2 * block_size>>>(
                n - block_size,
                thrust::raw_pointer_cast(data) + block_size,
                thrust::raw_pointer_cast(block_sums.data()) + 1
            );
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata, int block_size) {
            thrust::device_vector<int> input_data(idata, idata + n);

            timer().startGpuTimer();
            _scan(n, input_data.data(), block_size);
            timer().endGpuTimer();

            thrust::copy(input_data.begin(), input_data.end(), odata);
        }
    }
}

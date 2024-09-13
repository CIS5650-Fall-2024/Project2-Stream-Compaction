#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

template<typename T>
__device__ __host__ static inline void swap(T* &a, T* &b) {
  auto *const temp = a;
  a = b;
  b = temp;
}

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

            const int next_power_of_2 = 1 << (unsigned int) ceilf(log2f(n));

            // NOTE(rahul): I make the input and output offsets the reverse of what
            // they usually are so that when we execute the first for loop, we are correct
            // and when we exit, we can use output offset, which is semantically
            // correct.
            auto input_offset = 1;
            auto output_offset = 0;

            for (int d = 1; d < next_power_of_2; d <<= 1) {

                input_offset = 1 - input_offset;
                output_offset = 1 - output_offset;

                if (threadIdx.x >= d) {
                    shared[shared_index + output_offset] = shared[shared_index - d * 2 + input_offset] + shared[shared_index + input_offset]; 
                } else {
                    shared[shared_index + output_offset] = shared[shared_index + input_offset];
                }
                __syncthreads();
            }

            data[index] = shared[shared_index + output_offset];

            if (block_sums && threadIdx.x == blockDim.x - 1) {
                block_sums[blockIdx.x] = shared[shared_index + output_offset];
            }
        }

        __global__ void resolve_scan_blocks(int n, int *prescan_remainder, int *block_scan) {
            const auto index = blockIdx.x * blockDim.x + threadIdx.x;
            if (index >= n) return;
            prescan_remainder[index] += block_scan[blockIdx.x];
        }

        void naive_scan(int n, int *data) {
            const auto block_size = 128;
            const auto num_blocks = (n + block_size - 1) / block_size;

            // NOTE(rahul): this allocation takes 1 ms, we should ideally
            // take this out for performance metrics
            thrust::device_vector<int> block_sums(num_blocks);

            naive_scan_block<<<num_blocks, block_size, 2 * block_size * sizeof(int)>>>(
                n, data,
                thrust::raw_pointer_cast(block_sums.data())
            );

            if (num_blocks == 1) return;

            const auto block_sum_block_size = 128;
            const auto block_sum_num_blocks = (num_blocks + block_sum_block_size - 1) / block_sum_block_size;
            naive_scan_block<<<block_sum_num_blocks, block_sum_block_size, 2 * block_sum_block_size * sizeof(int)>>>(
                num_blocks,
                thrust::raw_pointer_cast(block_sums.data())
            );

            const auto final_scan_num_blocks = (n - 1) / block_size;
            resolve_scan_blocks<<<num_blocks, block_size, 2 * block_size>>>(
                n - block_size, data + block_size, thrust::raw_pointer_cast(block_sums.data())
            );
        }


        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            thrust::device_vector<int> input_data(idata, idata + n);

            timer().startGpuTimer();

            naive_scan(
                n, 
                thrust::raw_pointer_cast(input_data.data())
            );

            timer().endGpuTimer();

            thrust::copy(input_data.begin(), input_data.end(), odata);
        }
    }
}

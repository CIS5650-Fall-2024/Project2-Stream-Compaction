#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

#include <thrust/device_vector.h>

namespace StreamCompaction {
    namespace Efficient {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        __global__ void efficient_scan_block(int n, int *data, int *block_sums = nullptr) {
            extern __shared__ int shared[];

            const auto index = blockIdx.x * blockDim.x + threadIdx.x;
            if (index >= n) return;

            shared[threadIdx.x] = data[index];
            __syncthreads();

            auto offset = 1;
            for (auto d = blockDim.x >> 1; d > 0; d >>= 1, offset <<= 1) {
                if (threadIdx.x < d) {
                    shared[offset * (2 * threadIdx.x + 2) - 1] += shared[offset * (2 * threadIdx.x + 1) - 1];
                }
                __syncthreads();
            }

            if (threadIdx.x == 0) {
                if (block_sums) block_sums[blockIdx.x] = shared[blockDim.x - 1];
                shared[blockDim.x - 1] = 0;
            }
            __syncthreads();

            for (auto d = 1; d < blockDim.x; d <<= 1) {
                offset >>= 1;
                if (threadIdx.x < d) {
                    const auto left_child = offset * (2 * threadIdx.x + 1) - 1;
                    const auto parent = offset * (2 * threadIdx.x + 2) - 1;

                    const auto temp = shared[left_child];
                    shared[left_child] = shared[parent];
                    shared[parent] += temp;
                }
                __syncthreads();
            }

            data[index] = shared[threadIdx.x];
        }

        void _scan(int n, const thrust::device_ptr<int> &data, int block_size = 128) {
            const auto num_blocks = (n + block_size - 1) / block_size;

            thrust::device_vector<int> block_sums(num_blocks);

            efficient_scan_block<<<num_blocks, block_size, block_size * sizeof(int)>>>(
                n, thrust::raw_pointer_cast(data),
                thrust::raw_pointer_cast(block_sums.data())
            );

            if (num_blocks == 1) {
                return;
            } else if (num_blocks <= block_size) {
                const auto block_sum_num_blocks = (num_blocks + block_size - 1) / block_size;
                efficient_scan_block<<<block_sum_num_blocks, block_size, block_size * sizeof(int)>>>(
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
            thrust::device_vector<int> input_data(idata, idata + n);            
            thrust::device_vector<int> output_data(n);
            thrust::device_vector<int> is_not_zero(n);
            thrust::device_vector<int> indices;

            timer().startGpuTimer();
            
            const auto block_size = 128;
            const auto num_blocks = (n + block_size - 1) / block_size;

            Common::kernMapToBoolean<<<num_blocks, block_size>>>(
                n,
                thrust::raw_pointer_cast(is_not_zero.data()),
                thrust::raw_pointer_cast(input_data.data())
            );

            indices = is_not_zero;
            // do scan here
            _scan(n, indices.data());

            Common::kernScatter<<<num_blocks, block_size>>>(
                n, 
                thrust::raw_pointer_cast(output_data.data()), 
                thrust::raw_pointer_cast(input_data.data()),
                thrust::raw_pointer_cast(is_not_zero.data()),
                thrust::raw_pointer_cast(indices.data())
            );

            timer().endGpuTimer();

            thrust::copy(output_data.begin(), output_data.end(), odata);
            
            // NOTE(rahul): we add is_not_zero[n - 1] because we do 
            // an exclusive sum here. If the final element is not zero
            // we will not notice that we actually compact one more element
            // than the last element tells us.
            return indices[n - 1] + is_not_zero[n - 1];
        }
    }
}

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

        // declare the kernal function for up-sweeping
        __global__ void up_sweep(const int workload, const int level, int* outputs) {

            // compute the thread index
            const unsigned int index {blockIdx.x * blockDim.x + threadIdx.x};

            // avoid execution when the index is out of range
            if (index >= workload) {
                return;
            }

            // compute the offsets
            const int input_offset {((2 * index + 1) << level) - 1};
            const int output_offset {((2 * index + 2) << level) - 1};

            // compute the new value
            outputs[output_offset] += outputs[input_offset];
        }

        // declare the kernal function for up-sweeping
        __global__ void down_sweep(const int workload, const int level, int* outputs) {

            // compute the thread index
            const unsigned int index {blockIdx.x * blockDim.x + threadIdx.x};

            // avoid execution when the index is out of range
            if (index >= workload) {
                return;
            }

            // compute the offsets
            const int input_offset {((2 * index + 1) << level) - 1};
            const int output_offset {((2 * index + 2) << level) - 1};

            // store the current value at the input offset
            const int value {outputs[input_offset]};

            // overwrite the element at the input offset with the one at the output offset
            outputs[input_offset] = outputs[output_offset];

            // add the stored value to the element at the output offset
            outputs[output_offset] += value;
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {

            // declare the working buffer
            int* buffer;

            // calculate the next power of 2 of n
            const int count {1 << ilog2ceil(n)};

            // allocate the working buffer
            cudaMalloc(reinterpret_cast<void**>(&buffer), count * sizeof(int));

            // populate the working buffer with the input data
            cudaMemcpy(
                reinterpret_cast<void*>(buffer),
                reinterpret_cast<void*>(const_cast<int*>(idata)),
                n * sizeof(int), cudaMemcpyHostToDevice
            );

            // set the remaining elements in the working buffer to zeros
            cudaMemset(
                reinterpret_cast<void*>(buffer + n), 0,
                (count - n) * sizeof(int)
            );

            // start the timer after memory operations
            timer().startGpuTimer();
            
            // declare the block size
            const int block_size {32};

            // calculate the number of iterations needed to perform up-sweeping
            const int limit {ilog2ceil(n)};

            // perform up-sweeping "limit" number of times
            for (int level {0}; level < limit; level += 1) {

                // compute the workload
                const int workload {count >> (level + 1)};

                // perform up-sweeping
                up_sweep<<<workload / block_size + 1, block_size>>>(
                    workload, level, buffer
                );

                // wait until completion
                cudaDeviceSynchronize();
            }

            // set the last element of the working buffer to zero
            cudaMemset(
                reinterpret_cast<void*>(buffer + count - 1), 0,
                sizeof(int)
            );

            // perform down-sweeping "limit" number of times
            for (int level {limit - 1}; level >= 0; level -= 1) {

                // compute the workload
                const int workload {1 << (limit - level - 1)};

                // perform down-sweeping
                down_sweep<<<workload / block_size + 1, block_size>>>(
                    workload, level, buffer
                );

                // wait until completion
                cudaDeviceSynchronize();
            }
            
            // stop the timer before memory operations
            timer().endGpuTimer();

            // populate the output buffer with the working buffer
            cudaMemcpy(
                reinterpret_cast<void*>(odata),
                reinterpret_cast<void*>(buffer),
                n * sizeof(int), cudaMemcpyDeviceToHost
            );

            // free the working buffer
            cudaFree(reinterpret_cast<void*>(buffer));

            // avoid calling the original end timer function afterwards by returning
            return;

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

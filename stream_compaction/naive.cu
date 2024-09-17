#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"

namespace StreamCompaction {
    namespace Naive {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }
        
        // declare the kernal function for shifting and adding the elements
        __global__ void shift_and_add(const int count, const int offset,
                                      const int* inputs, int* outputs) {

            // compute the thread index
            const unsigned int index {blockIdx.x * blockDim.x + threadIdx.x};

            // avoid execution when the index is out of range
            if (index >= count) {
                return;
            }

            // perform shifting and adding after skipping "offset" number of elements
            if (index >= offset) {

                // read the input element stored in front of this element with the given offset
                const int input {inputs[index - offset]};

                // perform the adding and store the output
                outputs[index] = input + inputs[index];
            } else {

                // copy the data directly otherwise
                outputs[index] = inputs[index];
            }
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            
            // declare two buffers to swap and avoid conflicts
            int* buffers[2];

            // calculate the next power of 2 of n
            const int count {1 << ilog2ceil(n)};

            // allocate the buffers
            cudaMalloc(reinterpret_cast<void**>(&buffers[0]), count * sizeof(int));
            cudaMalloc(reinterpret_cast<void**>(&buffers[1]), count * sizeof(int));

            // populate the first buffer with the input data
            cudaMemcpy(
                reinterpret_cast<void*>(buffers[0]),
                reinterpret_cast<void*>(const_cast<int*>(idata)),
                n * sizeof(int), cudaMemcpyHostToDevice
            );

            // set the remaining elements in the first buffer to zeros
            cudaMemset(
                reinterpret_cast<void*>(buffers[0] + n), 0,
                (count - n) * sizeof(int)
            );

            // start the timer after memory operations
            timer().startGpuTimer();

            // declare the block size
            const int block_size {32};

            // calculate the number of iterations needed to perform shifting and adding
            const int limit {ilog2ceil(n)};

            // perform shifting and adding "limit" number of times
            for (int index {0}; index < limit; index += 1) {

                // calculate the offset using the shifting operator
                const int offset {1 << index};

                // perform shifting and adding
                shift_and_add<<<count / block_size + 1, block_size>>>(
                    n, offset, buffers[0], buffers[1]
                );

                // wait until completion
                cudaDeviceSynchronize();

                // swap the two buffers
                std::swap(buffers[0], buffers[1]);
            }
            
            // stop the timer before memory operations
            timer().endGpuTimer();

            // populate the output buffer with the first buffer with a single right shift
            cudaMemcpy(
                reinterpret_cast<void*>(odata + 1),
                reinterpret_cast<void*>(buffers[0]),
                (n - 1) * sizeof(int), cudaMemcpyDeviceToHost
            );

            // set the value for the first output
            odata[0] = 0;

            // free the buffers
            cudaFree(reinterpret_cast<void*>(buffers[0]));
            cudaFree(reinterpret_cast<void*>(buffers[1]));

            // avoid calling the original end timer function afterwards by returning
            return;

            timer().endGpuTimer();
        }
    }
}

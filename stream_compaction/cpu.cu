#include <cstdio>
#include "cpu.h"

#include "common.h"

namespace StreamCompaction {
    namespace CPU {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        /**
         * CPU scan (prefix sum).
         * For performance analysis, this is supposed to be a simple for loop.
         * (Optional) For better understanding before starting moving to GPU, you can simulate your GPU scan in this function first.
         */
        void scan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();
            
            // declare the variable that stores the sum
            int sum {0};

            // iterate through all indices
            for (int index {0}; index < n; index += 1) {

                // update the output
                odata[index] = sum;

                // increase the sum
                sum += idata[index];
            }

            timer().endCpuTimer();
        }

        /**
         * CPU stream compaction without using the scan function.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithoutScan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();
            
            // declare the number of elements in the output buffer
            int count {0};

            // iterate through all indices
            for (int index {0}; index < n; index += 1) {

                // read the next input from the buffer
                const int input {idata[index]};

                // store the input if it is a non-zero integer
                if (input != 0) {
                    odata[count] = input;

                    // increase the number of output elements
                    count += 1;
                }
            }

            // stop the timer
            timer().endCpuTimer();

            // return the number of output elements
            return count;

            timer().endCpuTimer();
            return -1;
        }

        /**
         * CPU stream compaction using scan and scatter, like the parallel version.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithScan(int n, int *odata, const int *idata) {
            
            // allocate a temporary buffer that stores the conditions
            int* condition_buffer {reinterpret_cast<int*>(std::malloc(sizeof(int) * n))};

            // abort when the allocation is not successful
            if (!condition_buffer) {
                std::abort();
            }

            // allocate another temporary buffer that stores the scan results
            int* index_buffer {reinterpret_cast<int*>(std::malloc(sizeof(int) * n))};

            // abort when the allocation is not successful
            if (!index_buffer) {
                std::abort();
            }

            // start the timer after memory operations
            timer().startCpuTimer();

            // iterate through all indices to compute the conditions
            for (int index {0}; index < n; index += 1) {

                // compute the condition based on the corresponding input value
                condition_buffer[index] = static_cast<int>(idata[index] != 0);
            }

            // declare the variable that stores the sum
            int sum {0};

            // iterate through all indices to perform a scan
            for (int index {0}; index < n; index += 1) {

                // update the output
                index_buffer[index] = sum;

                // increase the sum
                sum += condition_buffer[index];
            }

            // iterate through all indices to compute the final output
            for (int index {0}; index < n; index += 1) {

                // store the output only when the condition is true
                if (condition_buffer[index] == 1) {

                    // store the output at the correct index
                    odata[index_buffer[index]] = idata[index];
                }
            }

            // obtain the number of elements from the index buffer
            const int count {index_buffer[n - 1]};

            // stop the timer before memory operations
            timer().endCpuTimer();

            // free the allocated buffers
            std::free(condition_buffer);
            std::free(index_buffer);

            // return the number of output elements
            return count;

            timer().endCpuTimer();
            return -1;
        }
    }
}

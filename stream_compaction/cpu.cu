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
         * Exclusive prefix sum.
         */
        void scan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();

            // Put addition identity in first element.
            odata[0] = 0;
            // Serial version.
            for (int i = 1; i < n; ++i) {
                odata[i] = odata[i-1] + idata[i-1];
            }

            timer().endCpuTimer();
        }

        // CPU version of parallel algorithm. Incorrect.
        void scan2(int n, int *odata, const int *idata) {
            timer().startCpuTimer();

            // Each new iteration should update k in [2^d, ...] only.

            int *auxBuffer = (int *)malloc(n * sizeof(int));
            memcpy(auxBuffer, idata, n * sizeof(int));
            int *iterInput = auxBuffer;
            int *iterOutput = odata;

            odata[0] = 0;

            for (int d = 1; d <= ilog2ceil(n); ++d) {
                // At the beginning of each new iteration:
                //  - partial sums [0, 2^(d-1) - 1] are complete;
                //  - the rest are of the form x[k - 2^d - 1] + ... + x[k].
                for (int k = pow(2, d-1) + 1; k < n; ++k) {
                    int left = (k-1) - (int)pow(2, d-1);
                    iterOutput[k] = (left < 0 ? 0 : iterInput[left]) + iterInput[k-1];
                    // y[k] is now:
                    // = x[k] + x[k - 1] + x[k - 2] + ... + x[k - 4] + .... + x[k - 2^(d-1)]
                    // = x[max(0, k - 2^(d) + 1), k - 2^(d-1)] + x[k - 2^(d-1) + 1, k] 
                    // for d >= 1. 
                }
                 // Processing [2^d, n) completely before moving on to next d is equivalent
                // to waiting on a barrier for all threads to reach it, in the parallel case.

                memcpy(iterInput, iterOutput, n * sizeof(int));
            }

            if (iterInput != odata) {
                memcpy(odata, iterInput, n * sizeof(int));
            }

            free(auxBuffer);

            timer().endCpuTimer();
        }

        /**
         * CPU stream compaction without using the scan function.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithoutScan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();
            // TODO
            timer().endCpuTimer();
            return -1;
        }

        /**
         * CPU stream compaction using scan and scatter, like the parallel version.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithScan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();
            // TODO
            timer().endCpuTimer();
            return -1;
        }
    }
}

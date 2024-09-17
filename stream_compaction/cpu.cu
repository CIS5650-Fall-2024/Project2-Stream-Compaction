#include <cstdio>
#include "cpu.h"

#include "common.h"
#include <vector>

namespace StreamCompaction {
    namespace CPU {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        /**
         * CPU exlusive scan (prefix sum).
         * For performance analysis, this is supposed to be a simple for loop.
         * (Optional) For better understanding before starting moving to GPU, you can 
         * simulate your GPU scan in this function first.
         * 
         * If simulateGPUScan is true, the algorithm mimics the GPU parallel algorithm
         * of StreamCompaction::Naive::scan to some extent; differences lie around how
         * the GPU version deals with arbitrary length inputs across more than 1 block.
         */
        void scan(int n, int *odata, const int *idata, bool simulateGPUScan) {
            timer().startCpuTimer();

            if (simulateGPUScan) {
                scanExclusiveSimulateGPU(n, odata, idata);
            } else {
                scanExclusiveSerial(n, odata, idata);
            }

            timer().endCpuTimer();
        }

        inline void scanExclusiveSerial(int n, int *odata, const int *idata) {
            // Put addition identity in first element.
            odata[0] = 0;
            // Serial version.
            for (int i = 1; i < n; ++i) {
                odata[i] = odata[i-1] + idata[i-1];
            }
        }

        // CPU version of parallel algorithm. Mimics the GPU parallel algorithm in 
        // StreamCompaction::Naive::scan to some extent. The intention is not to be
        // efficient, but to help understand the parallel algorithm.
        inline void scanExclusiveSimulateGPU(int n, int *odata, const int *idata) {
            // For each depth d, iterInput is read from and iterOutput is written to
            // and then swapped.
            std::vector<int> auxBuffer(n);
            int *iterInput = auxBuffer.data();
            int *iterOutput = odata;

            // Put addition identity in first element.
            iterInput[0] = 0;

            // Copy input data to iterInput, shifting by 1. This effectively turns
            // an inclusive scan into an exclusive scan.
            for (int k = 1; k < n; ++k) {
                iterInput[k] = idata[k-1]; 
            }

            for (int d = 1; d <= ilog2ceil(n); ++d) {
                // Elements to be added are this much apart.
                int delta = 1 << (d-1);

                // At the beginning of each new iteration:
                //  - partial sums [0, 2^(d-1) - 1] are complete;
                //  - the rest are of the form x[k - 2^d - 1] + ... + x[k].
                for (int k = 0; k < n; ++k) {
                    // Each new iteration should update k in [2^d-1, ...] only.
                    if (k > delta) {
                        // Note that if k = delta, then iterInput[k - delta] = 0, so 
                        // that's handled by the other case.
                        iterOutput[k] = iterInput[k - delta] + iterInput[k];
                    } else {
                        iterOutput[k] = iterInput[k];
                    }

                    // y[k] is now:
                    // = x[k] + x[k - 1] + x[k - 2] + ... + x[k - 4] + .... + x[k - 2^(d-1)]
                    // = x[max(0, k - 2^(d) + 1), k - 2^(d-1)] + x[k - 2^(d-1) + 1, k] 
                    // for d >= 1. 
                }

                // Processing [2^d, n) completely before moving on to next d is equivalent
                // to waiting on a barrier for all threads to reach it, in the parallel case.

                std::swap(iterInput, iterOutput);
            }

            // Depending on the number of iterations, the final result (iterInput) may already
            // be in odata.
            if (iterInput != odata) {
                memcpy(odata, iterInput, n * sizeof(int));
            }
        }

        /**
         * CPU stream compaction without using the scan function.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithoutScan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();
            
            int nonzeroCount = 0;

            for (int i = 0; i < n; ++i) {
                if (idata[i] != 0) {
                    odata[nonzeroCount++] = idata[i];
                }
            }

            timer().endCpuTimer();

            return nonzeroCount;
        }

        /**
         * CPU stream compaction using scan and scatter, like the parallel version.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithScan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();

            // Identify non-zero elements by marking them with 1 in the mask.
            int *nonzeroMask = new int[n];
            for (int i = 0; i < n; ++i) {
                nonzeroMask[i] = (idata[i] != 0) ? 1 : 0;
            }

            // Exclusive scan the mask.
            int *nonzeroMaskPrefixSum = new int[n];
            scanExclusiveSerial(n, nonzeroMaskPrefixSum, nonzeroMask);

            // Scatter the non-zero elements.
            int nonzeroCount = 0;
            for (int i = 0; i < n; ++i) {
                if (nonzeroMask[i]) {
                    odata[nonzeroMaskPrefixSum[i]] = idata[i];
                    nonzeroCount++;
                }
            }

            free(nonzeroMaskPrefixSum);
            free(nonzeroMask);

            timer().endCpuTimer();

            return nonzeroCount;
        }
    }
}

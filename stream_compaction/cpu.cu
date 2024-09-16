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
            odata[0] = 0; // Exclusive scan
            for (int i = 1; i < n; i++) {
                odata[i] = odata[i - 1] + idata[i - 1];
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
            int count = 0;
            for (int i = 0; i < n; i++) {
                if (idata[i] != 0) {
                    odata[count++] = idata[i];
                }
            }
            timer().endCpuTimer();
            return count;
        }

        /**
         * CPU stream compaction using scan and scatter, like the parallel version.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithScan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();
            int* flags = new int[n];
            int* scanned = new int[n];

            // Map input to flags (0 or 1)
            for (int i = 0; i < n; i++) {
                flags[i] = (idata[i] != 0) ? 1 : 0;
            }

            // Perform scan on flags
            scan(n, scanned, flags);

            // Scatter non-zero elements
            int count = 0;
            for (int i = 0; i < n; i++) {
                if (flags[i]) {
                    odata[scanned[i]] = idata[i];
                    count++;
                }
            }

            delete[] flags;
            delete[] scanned;
            timer().endCpuTimer();
            return count;
        }
    }
}

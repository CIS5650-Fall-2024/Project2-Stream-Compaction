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
        void scan(int n, int* odata, const int* idata) {
            timer().startCpuTimer();
            // TODO

            if (n == 0) {
                timer().endCpuTimer();
                return;
            }

            odata[0] = 0;
            for (int i = 1; i < n; ++i) {
                odata[i] = odata[i - 1] + idata[i - 1];
            }

            timer().endCpuTimer();
        }

        /**
         * CPU stream compaction without using the scan function.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithoutScan(int n, int* odata, const int* idata) {
            timer().startCpuTimer();
            // TODO
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
        int compactWithScan(int n, int* odata, const int* idata) {
            int* bools = new int[n];
            int* indices = new int[n];

            // Map: mark 1 if non-zero, else 0
            for (int i = 0; i < n; i++) {
                bools[i] = (idata[i] != 0) ? 1 : 0;
            }

            // Scan the bools to get indices
            scan(n, indices, bools);

            // Scatter: place valid elements in correct position
            int count = 0;
            for (int i = 0; i < n; i++) {
                if (bools[i] == 1) {
                    odata[indices[i]] = idata[i];
                    count++;
                }
            }

            delete[] bools;
            delete[] indices;

            return count;
        }
    }
}

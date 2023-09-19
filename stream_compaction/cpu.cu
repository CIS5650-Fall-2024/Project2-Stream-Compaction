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
            for (int i = 0; i < n; i++) {
                odata[i] = idata[i];
                if (i > 0) {
                    odata[i] += odata[i - 1];
                }
            }
            timer().endCpuTimer();
        }

        void _scan_no_timer(int n, int *odata, const int *idata) {
            for (int i = 0; i < n; i++) {
                odata[i] = idata[i];
                if (i > 0) {
                    odata[i] += odata[i - 1];
                }
            }
        }

        /**
         * CPU stream compaction without using the scan function.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithoutScan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();
            int numElements = 0;
            for (int i = 0; i < n; i++) {
                if (idata[i] != 0) {
                    odata[numElements] = idata[i];
                    numElements++;
                }
            }
            timer().endCpuTimer();
            return numElements;
        }

        /**
         * CPU stream compaction using scan and scatter, like the parallel version.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithScan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();
            int* bools = new int[n];
            for (int i = 0; i < n; i++) {
                bools[i] = (idata[i] != 0) ? 1 : 0;
            }
            int* scanned = new int[n];
            _scan_no_timer(n, scanned, bools);
            // convert to exclusive scan
            for (int i = n - 1; i > 0; i--) {
                scanned[i] = scanned[i - 1];
            }
            scanned[0] = 0;
            int numElements = 0;
            for (int i = 0; i < n; i++) {
                if (bools[i] != 0) {
                    odata[scanned[i]] = idata[i];
                    numElements++;
                }
            }

            delete[] bools;
            delete[] scanned;
            timer().endCpuTimer();
            return numElements;
        }
    }
}

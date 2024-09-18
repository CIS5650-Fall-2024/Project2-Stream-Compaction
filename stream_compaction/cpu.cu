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

            odata[0] = 0;
            for (int i = 1; i < n; i++) {
                odata[i] = idata[i - 1] + odata[i - 1];
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
                    odata[count] = idata[i];
                    count++;
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
            int* t = new int[n];
            int* scanned = new int[n];

            // map to 0 and 1 array
            for (int i = 0; i < n; i++) {
                if (idata[i] != 0) {
                    t[i] = 1;
                }
                else {
                    t[i] = 0;
                }
            }

            // scan
            scanned[0] = 0;
            for (int i = 1; i < n; i++) {
              scanned[i] = t[i - 1] + scanned[i - 1];
            }

            // scatter
            int count = 0;
            for (int i = 0; i < n; i++) {
              if (t[i] != 0) {
                odata[scanned[i]] = idata[i];
                count = scanned[i];
              }
            }

            timer().endCpuTimer();
            delete[] t;
            delete[] scanned;
            return count + 1;
        }
    }
}

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
            for (int k = 1; k < n; ++k) {
                odata[k] = odata[k - 1] + idata[k - 1];
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

            int j = 0;
            for (int i = 0; i < n; ++i) {
                if (idata[i] != 0) {
                    odata[j] = idata[i];
                    ++j;
                }
            }

            timer().endCpuTimer();
            return j;
        }

        /**
         * CPU stream compaction using scan and scatter, like the parallel version.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithScan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();

            // temp array
            int* temp = new int[n];
            for (int i = 0; i < n; ++i) {
                temp[i] = (idata[i] == 0) ? 0 : 1;
            }

            // scan
            odata[0] = 0;
            for (int k = 1; k < n; ++k) {
                odata[k] = odata[k - 1] + temp[k - 1];
            }

            int numRemaining = odata[n - 1];

            // scatter
            for (int i = 0; i < n; ++i) {
                if (temp[i] == 1) {
                    odata[odata[i]] = idata[i];
                }
            }

            delete[] temp;

            timer().endCpuTimer();
            return numRemaining;
        }
    }
}

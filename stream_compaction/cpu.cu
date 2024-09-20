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
            int j = 0;
            for (int i = 0; i < n; i++) {
                if (idata[i]) {
                    odata[j] = idata[i];
                    j++; 
                }
            }
            timer().endCpuTimer();
            return (j) ? j : -1;
        }

        /**
         * CPU stream compaction using scan and scatter, like the parallel version.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithScan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();
            // map the input array to an array of 0s and 1s
            int* temp = new int[n];
            for (int i = 0; i < n; i++) {
                temp[i] = idata[i] == 0 ? 0 : 1;
            }

            int* scanOutput = new int[n];
            scanOutput[0] = 0;

            // scan the temp array
            for (int i = 1; i < n; i++) {
                scanOutput[i] = scanOutput[i - 1] + temp[i - 1];
            }

            int compactLen = scanOutput[n - 1];

            for (int i = 0; i < n; i++) {
                if (temp[i]) {
                    odata[scanOutput[i]] = idata[i];
                }
            }
            timer().endCpuTimer();
            return (compactLen) ? compactLen : -1;
        }
    }
}

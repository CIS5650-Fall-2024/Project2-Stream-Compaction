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
                odata[i] = odata[i - 1] + idata[i - 1];
            }

            timer().endCpuTimer();
        }

        // Use this in compactWithScan so that we won't have "CPU timer already started" bug
        void scan_without_timer(int n, int *odata, const int *idata) {
            odata[0] = 0;
            for (int i = 1; i < n; i++) {
                odata[i] = odata[i - 1] + idata[i - 1];
            }
        }

        /**
         * CPU stream compaction without using the scan function.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithoutScan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();

            int cur_out_idx = 0;
            for (int i = 0; i < n; i++) {
                if (idata[i] != 0) {
                    odata[cur_out_idx] = idata[i];
                    cur_out_idx++;
                }
            }

            timer().endCpuTimer();
            return cur_out_idx;
        }

        /**
         * CPU stream compaction using scan and scatter, like the parallel version.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithScan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();

            // Step 1: Map idata to a 0/1 array
            int *bools = new int[n]; // Array to hold the 0s and 1s
            for (int i = 0; i < n; i++) {
                bools[i] = (idata[i] != 0) ? 1 : 0;
            }

            // Step 2: Perform the scan using the existing scan function
            int *scanResult = new int[n]; // Array to hold the scan result
            scan_without_timer(n, scanResult, bools);   // Perform the scan on the bools array
            
            // Step 3: Scatter non-zero elements into odata
            int count = 0;
            for (int i = 0; i < n; i++) {
                if (bools[i] == 1) {
                    odata[scanResult[i]] = idata[i];
                    count++;
                }
            }
            
            timer().endCpuTimer();
            return count;
        }
    }
}

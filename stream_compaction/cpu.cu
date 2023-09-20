#include <cstdio>
#include "cpu.h"

#include "common.h"

namespace StreamCompaction {
    namespace CPU {
        bool disableScanTimer = false;
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
            if (!disableScanTimer) timer().startCpuTimer();
            // TODO
            odata[0] = 0;
            for (int i = 1; i < n; i++) {
                odata[i] = odata[i - 1] + idata[i - 1];
            }
            if (!disableScanTimer) timer().endCpuTimer();
        }

        /**
         * CPU stream compaction without using the scan function.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithoutScan(int n, int *odata, const int *idata) {
            int count = 0;
            timer().startCpuTimer();
            // TODO
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
            int* tmp = new int[n];
            disableScanTimer = true;
            timer().startCpuTimer();
            // TODO
            for (int i = 0; i < n; i++) {
                tmp[i] = idata[i] == 0 ? 0 : 1;
            }
            scan(n, odata, tmp);
            for (int i = 0; i < n; i++) {
                if (tmp[i] != 0) {
                    // odata[i] <= i, so there is no race condition
                    odata[odata[i]] = idata[i];
                }
            }
            timer().endCpuTimer();
            disableScanTimer = false;
            int count = odata[n - 1] + tmp[n - 1];
            delete[] tmp;
            return count;
        }
    }
}

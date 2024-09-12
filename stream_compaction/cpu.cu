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
        void scan(int n, int *odata, const int *idata, bool time) {
            if (time)
                timer().startCpuTimer();
            // TODO
            odata[0] = 0;
            for (int i = 1; i < n; i++) {
                odata[i] = odata[i - 1] + idata[i - 1];
            }
            if (time)
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
            int index = 0;
            for (int i = 0; i < n; i++) {
                if (idata[i]) {
                    odata[index] = idata[i];
                    index++;
                }
            }
            timer().endCpuTimer();
            return index;
        }

        /**
         * CPU stream compaction using scan and scatter, like the parallel version.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithScan(int n, int *odata, const int *idata) {
            // TODO
            int *bidata = new int[n];
            int *bodata = new int[n];

            timer().startCpuTimer();

            // map
            for (int i = 0; i < n; i++) {
                bidata[i] = idata[i] ? 1 : 0;
            }

            // scan
            scan(n, bodata, bidata, false);

            // scatter
            for (int i = 0; i < n; i++) {
                if (bidata[i]) {
                    odata[bodata[i]] = idata[i];
                }
            }

            delete[] bidata;
            delete[] bodata;

            timer().endCpuTimer();
            return bodata[n - 1];
        }
    }
}

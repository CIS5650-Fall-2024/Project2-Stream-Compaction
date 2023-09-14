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
            // TODO
            for (int i = 0; i < n; ++i) {
                for (int j = 0; j < i /* exclusive prefix sum */; ++j) {
                    odata[i] += idata[j];
                }
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
            // TODO
            int oPtr = 0;
            for (int i = 0; i < n; ++i) {
                if (idata[i]) odata[oPtr++] = idata[i];
            }
            timer().endCpuTimer();
            return oPtr;
        }

        /**
         * CPU stream compaction using scan and scatter, like the parallel version.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithScan(int n, int *odata, const int *idata) {
            int* odata_tmp = new int[n];
            timer().startCpuTimer();
            for (int i = 0; i < n; ++i) {
                odata_tmp[i] = !(!idata[i]);
                if (i) odata_tmp[i] += odata_tmp[i - 1];
                if (idata[i]) {
                    odata[odata_tmp[i] - 1] = idata[i];
                }
            }
            int oSize = odata_tmp[n-1];
            timer().endCpuTimer();
            delete [] odata_tmp;

            return oSize;
        }
    }
}

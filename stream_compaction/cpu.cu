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
            // DONE
            odata[0] = 0;
            for (int i = 1; i < n; i++)
            {
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
            // DONE
            int oi = 0, ii = 0;
            while (ii < n) {
                if (idata[ii] != 0) {
                    odata[oi++] = idata[ii];
                }
                ii++;
            }
            timer().endCpuTimer();
            return oi;
        }

        /**
         * CPU stream compaction using scan and scatter, like the parallel version.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithScan(int n, int* odata, const int* idata) {
            int* notZero = new int[n];
            timer().startCpuTimer();
            // DONE
            notZero[0] = 0;
            for (int i = 0; i < n - 1; i++)
            {
                notZero[i + 1] = notZero[i] + (idata[i] == 0 ? 0 : 1);
            }
            int num = notZero[n - 1];
            for (int i = 0; i < n; i++)
            {
                if (idata[i] != 0) {
                    odata[notZero[i]] = idata[i];
                }
            }
            timer().endCpuTimer();
            delete[]notZero;
            return num;
        }
    }
}

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
            // TODO
            int cnt = 0;
            odata[0] = 0;
            for (int i = 0; i < n; i++) {
                if(idata[i] != 0)
                {
                    odata[cnt] = idata[i];
                    cnt++;
                }
            }
            timer().endCpuTimer();
            return cnt;
        }

        /**
         * CPU stream compaction using scan and scatter, like the parallel version.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithScan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();
            // TODO

            // temporary array
            int *temp = new int[n];
            for (int i = 0; i < n; i++) {
                temp[i] = (idata[i] != 0);
            }
            int *temp2 = new int[n];

            // scan
            temp2[0] = 0;
            for (int i = 1; i < n; i++) {
                temp2[i] = temp[i - 1] + temp2[i - 1];
            }
            int cnt = temp2[n - 1];

            // scatter
            for (int i = 0; i < n; i++) {
                if (temp[i] > 0)
                {
                    odata[temp2[i]] = idata[i];
                }
            }

            timer().endCpuTimer();
            delete[] temp;
            delete[] temp2;
            return cnt;
        }
    }
}

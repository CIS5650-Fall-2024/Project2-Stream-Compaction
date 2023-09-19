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
            for (int i = 0; i < n-1; i++) {
                odata[i+1] = idata[i]+odata[i];
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
            int j = 0;
            for (int i = 0; i < n; i++) {
                if (idata[i] != 0) {
                    odata[j++] = idata[i];
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
            // TODO

            int* temp=(int*)malloc(n * sizeof(int));
            int* temp2 = (int*)malloc(n * sizeof(int));

            //mapping
            for (int i = 0; i < n; i++) {
                if (idata[i] == 0) {
                    temp[i] = 0;
                }
                else {
                    temp[i] = 1;
                }
            }
            //scan
            temp2[0] = 0;
            for (int i = 0; i < n - 1; i++) {
                temp2[i + 1] = temp[i] + temp2[i];
            }
            //scatter
            for (int i = 0; i < n; i++) {
                if (temp[i]==1) {
                    odata[temp2[i]] = idata[i];
                }
            }
            int cnt = temp2[n - 1];
            free(temp);
            free(temp2);
            timer().endCpuTimer();
            return cnt;
        }
    }
}

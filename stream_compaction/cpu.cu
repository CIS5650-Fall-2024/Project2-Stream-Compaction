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
        //cited Lecture slide
        void scan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();
            // TODO
            odata[0] = 0;
            for (int k = 1; k < n; ++k) {
                odata[k] = odata[k - 1] + idata[k-1];
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
            for (int k = 0; k < n; k++) {
                if (idata[k] != 0) {
                    odata[j] = idata[k];
                    j++;
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
            int* temp = new int[n];
            int* scan = new int[n];
            int k = 0;
            int j = 1;
            int oindex = 0;
            for (int i = 0; i < n; i++) {
                if (idata[i] == 0) {
                    temp[i] = 0;
                }
                else {
                    temp[i] = 1;
                }

            }

            for (int i = 0; i < n; i++) {
                scan[i] = k;
                if (temp[i] == 1) {
                    k++;
                }
                
                
            }



            for (int i = 0; i < n; i++) {
                if (scan[i] == j) {
                    odata[oindex] = idata[i - 1];
                    oindex += 1;
                    j += 1;
                }
            }
            delete[] temp;
            timer().endCpuTimer();
            return oindex;
        }
    }
}

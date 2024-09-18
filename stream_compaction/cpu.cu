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
         * Standard sort
         * (for parallel radix comparison)
         */
        void sort(int n, int* odata, const int* idata) {
            memcpy(odata, idata, n * sizeof(int));

            timer().startCpuTimer();
            std::sort(odata, odata + n);
            timer().endCpuTimer();
        }

        /**
         * CPU scan (prefix sum).
         * For performance analysis, this is supposed to be a simple for loop.
         * (Optional) For better understanding before starting moving to GPU, you can simulate your GPU scan in this function first.
         */
        void scan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();
            
            odata[0] = 0;
            for (int i = 0; i < n - 1; i++) {
                odata[i + 1] = odata[i] + idata[i];
            }

            timer().endCpuTimer();
        }

        /**
         * CPU stream compaction without using the scan function.
         * remove 0s from an array of ints
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithoutScan(int n, int *odata, const int *idata) {
            int idx = 0;
            timer().startCpuTimer();
           
            for (int i = 0; i < n; i++) {
                if (idata[i] != 0) {
                    odata[idx] = idata[i];
                    idx++;
                }
            }

            timer().endCpuTimer();
            return idx;
        }

        /**
         * CPU stream compaction using scan and scatter, like the parallel version.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithScan(int n, int *odata, const int *idata) {
            int* tmp = new int[n];
            int* scanned = new int[n];
            timer().startCpuTimer();
            
            // temporary array
            for (int i = 0; i < n; i++) {
                tmp[i] = (idata[i] == 0) ? 0 : 1;
            }
            
            // copied scan function (to not call timer twice)
            scanned[0] = 0;
            for (int i = 0; i < n - 1; i++) {
                scanned[i + 1] = scanned[i] + tmp[i];
            }

            // scatter
            for (int i = 0; i < n; i++) {
                if (tmp[i] == 1) {
                    odata[scanned[i]] = idata[i];
                }
            }

            timer().endCpuTimer();
            int num = scanned[n - 1] + tmp[n - 1];
            delete[] tmp;
            delete[] scanned;
            return num;
        }
    }
}

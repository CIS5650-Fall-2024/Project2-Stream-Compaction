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
            int sum = 0;
            for (int i = 0; i < n; i++) {
                odata[i] = sum;
                sum += idata[i];
                
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
            int length = 0;
            for (int i = 0; i < n; i++) {
                if (idata[i] != 0) {
                    odata[length] = idata[i];
                    length++;
                }
            }
            timer().endCpuTimer();
            return length;
        }

        /**
         * CPU stream compaction using scan and scatter, like the parallel version.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithScan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();
            int* temp = new int[n];
            int* tempScanned = new int[n];
            for (int i = 0; i < n; i++) {
                idata[i] > 0 ? temp[i] = 1 : temp[i] = 0;
            }

            int sum = 0;
            for (int i = 0; i < n; i++) {
                tempScanned[i] = sum;
                sum += temp[i];
            }

            int count = 0;
            for (int i = 0; i < n; i++) {
                if (temp[i] == 1) {
                    odata[tempScanned[i]] = idata[i];
                    count++;
                }
            }
            timer().endCpuTimer();
            return count;
        }
    }
}

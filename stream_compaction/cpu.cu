#include <cstdio>
#include "cpu.h"
#include <iostream>

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
            // TODO: traverse all elements and record the exclusive prefix sum to odata.
            odata[0] = 0;
            for (int i = 1; i < n; i++) {
                odata[i] = odata[i-1] + idata[i-1];
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
            // TODO: traverse all elements and copy to odata without 0's.
            int count = 0;
            for (int i = 0; i < n; i++) {
                if (idata[i] != 0) {
                    odata[count] = idata[i];
                    count++;
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
            int* scanResult = new int[n];
            int* bools = new int[n];
            timer().startCpuTimer();
            // TODO
            //std::cout << "start traverse and build temp arr" << std::endl;
            int sum = 0;
            for (int i = 0; i < n; i++) {
                bools[i] = idata[i] == 0 ? 0 : 1;
                scanResult[i] = sum;
                sum += bools[i];
            }
            for (int i = 0; i < n; i++) {
                if (bools[i] == 1) {
                    odata[scanResult[i]] = idata[i];
                }
            }
            timer().endCpuTimer();
            int count = scanResult[n - 1] + bools[n - 1];
            delete[] scanResult;
            return count;
        }
    }
}

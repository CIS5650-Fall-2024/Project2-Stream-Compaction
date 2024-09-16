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
            int total = 0;
            for (int i = 0; i < n; i++) {
                odata[i] = total;
                total += idata[i];
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
            int numElements = 0;
            for (int i = 0; i < n; i++) {
                int value = idata[i];
                if (value != 0) {
                    odata[numElements++] = value;
                }
            }
            timer().endCpuTimer();
            return numElements;
        }

        int scatter(int n, int *odata, const int *idata, const int *bools, const int *indices) {
            int numElements = 0;
            for (int i = 0; i < n; i++) {
                if (bools[i] == 1) {
                    odata[indices[i]] = idata[i];
                    numElements++;
                }
            }
            return numElements;
        }

        /**
         * CPU stream compaction using scan and scatter, like the parallel version.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithScan(int n, int *odata, const int *idata) {
            int* bools = new int[n];
            int* scanResult = new int[n];
            timer().startCpuTimer();
            for (int i = 0; i < n; i++) {
                bools[i] = idata[i] != 0 ? 1 : 0;
            }
            scan(n, scanResult, bools);
            int numElements = scatter(n, odata, idata, bools, scanResult);
            timer().endCpuTimer();
            delete[] bools;
            delete[] scanResult;
            return numElements;
        }
    }
}

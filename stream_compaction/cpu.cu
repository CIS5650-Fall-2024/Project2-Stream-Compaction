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
            
            // exclusive scan
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
            
            int cur = 0;
            for (int i = 0; i < n; i++) {
                int val = idata[i];
                if (val != 0) odata[cur++] = val;
            }

            timer().endCpuTimer();
            
            return cur;
        }

        /**
         * CPU stream compaction using scan and scatter, like the parallel version.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithScan(int n, int *odata, const int *idata) {
            int* tmp = new int[n];
            
            timer().startCpuTimer();
            
            // map
            for (int i = 0; i < n; i++) {
                tmp[i] = idata[i] ? 1 : 0;
            }

            // scan
            int sum = 0;
            for (int i = 0; i < n; i++) {
                odata[i] = sum;
                sum += tmp[i];
            }

            //scatter
            int count = tmp[n - 1] + odata[n - 1];
            for (int i = 0; i < n ; i++) {
                if (tmp[i]) odata[odata[i]] = idata[i];
            }

            timer().endCpuTimer();
            
            delete[] tmp;
            return count;
        }
    }
}

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
            int running_sum = 0;
            for (int i = 0; i < n; i++) {
              odata[i] = running_sum;
              running_sum += idata[i];
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
            int l_ptr = 0;
            for (int r_ptr = 0; r_ptr < n; r_ptr++) {
              if (idata[r_ptr] != 0) {
                odata[l_ptr] = idata[r_ptr];
                l_ptr++;
              }
            }
            timer().endCpuTimer();
            return l_ptr;
        }

        /**
         * CPU stream compaction using scan and scatter, like the parallel version.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithScan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();
            // TODO
            int *mask = new int[n];
            for (int i = 0; i < n; i++) {
              mask[i] = idata[i] == 0 ? 0 : 1;
            }
            scan(n, odata, mask);
            int remain_cnt = 0;
            for (int i = 0; i < n; i++) {
              if (mask[i] == 1) {
                odata[odata[i]] = idata[i];
                remain_cnt++;
              }
            }
            delete[] mask;
            timer().endCpuTimer();
            return remain_cnt;
        }
    }
}

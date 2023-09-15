#include <cstdio>
#include <vector>
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
            if (n <= 0) return;
            timer().startCpuTimer();
            odata[0] = 0;
            for (int i = 1; i < n; ++i)
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
        int compactWithoutScan(int n, int *odata, const int *idata) {
            int index = 0;
            timer().startCpuTimer();
            for (int i = 0; i < n; ++i)
            {
                if (idata[i] != 0)
                {
                    odata[index++] = idata[i];
                }
            }
            timer().endCpuTimer();
            return index;
        }

        /**
         * CPU stream compaction using scan and scatter, like the parallel version.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithScan(int n, int *odata, const int *idata) {
            std::vector<int> label(n);
            std::vector<int> pre_sum_exclusive(n);

            timer().startCpuTimer();
            // label each item with 0/1
            for (int i = 0; i < n; ++i)
            {
                label[i] = (idata[i] != 0 ? 1 : 0);
            }
            // exclusive scan
            pre_sum_exclusive[0] = 0;
            for (int i = 1; i < n; ++i)
            {
                pre_sum_exclusive[i] = pre_sum_exclusive[i - 1] + label[i - 1];
            }
            // scatter
            for (int i = 0; i < n; ++i)
            {
                if (idata[i] != 0)
                {
                    odata[pre_sum_exclusive[i]] = idata[i];
                }
            }

            timer().endCpuTimer();
            return pre_sum_exclusive[n - 1];
        }
    }
}

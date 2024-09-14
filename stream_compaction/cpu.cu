#include <cstdio>
#include "cpu.h"
#include <vector>

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
        void scan(int n, int *odata, const int *idata, bool time) {
            if (time) timer().startCpuTimer();
            odata[0] = 0;
            for (auto i = 1; i < n; i++) {
                odata[i] = odata[i - 1] + idata[i - 1];
            }
            if (time) timer().endCpuTimer();
        }

        /**
         * CPU stream compaction without using the scan function.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithoutScan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();
            auto j = 0;
            for (auto i = 0; i < n; i++) {
                if (idata[i] != 0) {
                    odata[j] = idata[i];
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
            std::vector<int> is_not_zero(n);
            for (auto i = 0; i < n; i++) {
                is_not_zero[i] = idata[i] != 0;
            }
            
            // NOTE(rahul): we reuse the odata buffer here.
            scan(n, odata, is_not_zero.data(), false);

            const auto num_elements = odata[n - 1] + is_not_zero[n - 1];

            // scatter 
            // NOTE(rahul): this works because we know that odata[i]
            // will always be <= i, This only works in the sequential
            // case.
            for (auto i = 0; i < n; i++) {
                if (is_not_zero[i] == 1) {
                    odata[odata[i]] = idata[i];
                }
            }

            timer().endCpuTimer();
            return num_elements;
        }
    }
}

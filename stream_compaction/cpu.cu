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
            odata[0] = idata[0];
            for (auto i = 1; i < n; i++) {
                odata[i] = odata[i - 1] + idata[i];
            }
            if (time) timer().endCpuTimer();
        }

        static inline void inclusive_to_exclusive(int n, int *odata, const int *idata) {
            // NOTE(rahul): because odata can alias to idata, we need to 
            // iterate over the array backwards
            for (auto i = n - 1; i > 0; i--) {
                odata[i] = idata[i - 1];
            }
            odata[0] = 0;
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
            return j - 1;
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
            inclusive_to_exclusive(n, odata, odata);

            const auto num_elements = odata[n - 1] - 1;

            // scatter 
            // NOTE(rahul): this works because we know that odata[i]
            // will always be <= i,
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

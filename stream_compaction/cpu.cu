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

            odata[0] = 0;
            for (int k = 1; k < n; ++k)
            {
                odata[k] = odata[k - 1] + idata[k - 1];
            }

            timer().endCpuTimer();
        }

        /**
         * CPU stream compaction without using the scan function.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithoutScan(int n, int *odata, const int *idata) {
            int size = 0;

            timer().startCpuTimer();
            
            for (int k = 0; k < n; ++k)
            {
                if (idata[k] != 0)
                {
                    odata[size] = idata[k];
                    ++size;
                }
            }

            timer().endCpuTimer();
            return size;
        }

        /**
         * CPU stream compaction using scan and scatter, like the parallel version.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithScan(int n, int *odata, const int *idata) {
            int* indices = new int[n];
            int size = 0;

            timer().startCpuTimer();

            // Compute the temporary array of pass/fail checks
            for (int k = 0; k < n; ++k)
            {
                odata[k] = idata[k] != 0;
            }
            
            // Scan the temporary array
            indices[0] = 0;
            for (int k = 1; k < n; ++k)
            {
                indices[k] = indices[k - 1] + odata[k - 1];
            }

            // Scatter based on the found indices
            for (int k = 0; k < n; ++k)
            {
                if (odata[k] != 0)
                {
                    odata[indices[k]] = idata[k];
                    ++size;
                }
            }

            timer().endCpuTimer();

            delete[](indices);

            return size;
        }
    }
}

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
            // exclusive scan!
            // odata[0] = 0;   // don't need this line since odata is already filled with 0s
            for (int i = 1; i < n; i++)
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
            timer().startCpuTimer();
            int o = 0;
            for (int i = 0; i < n; i++)
            {
                if (idata[i] != 0)
                {
                    odata[o++] = idata[i];
                }
            }
            timer().endCpuTimer();
            return o;
        }

        /**
         * CPU stream compaction using scan and scatter, like the parallel version.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithScan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();
            // generate temp array with 1 where element != 0 and 0 otherwise
            int* temp = new int[n];
            for (int i = 0; i < n; i++)
            {
                temp[i] = idata[i] == 0 ? 0 : 1;
            }

            // run scan, copy the code from scan() because we can't use scan() here
            // because scan() runs its own timer and running a timer while its already
            // running throws an exception. :)
            //scan(n, odata, temp);
            for (int i = 1; i < n; i++)
            {
                odata[i] = odata[i - 1] + temp[i - 1];
            }

            // scatter
            for (int i = 0; i < n; i++)
            {
                if (temp[i] == 1)
                {
                    odata[odata[i]] = idata[i];
                }
            }

            delete[] temp;
            timer().endCpuTimer();
            return odata[n - 1];
        }
    }
}

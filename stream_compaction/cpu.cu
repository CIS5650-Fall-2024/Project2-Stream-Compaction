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

        void map(int n, int* odata)
        {
            for (int i = 0; i < n; i++)
            {
                if (odata[i] != 0)
                {
                    odata[i] = 1;
                }
            }
        }

        /**
         * CPU scan (prefix sum).
         * For performance analysis, this is supposed to be a simple for loop.
         * (Optional) For better understanding before starting moving to GPU, you can simulate your GPU scan in this function first.
         */
        void scan(int n, int *odata, const int *idata) {
            // TODO
            std::copy(idata, idata + n - 1, &(odata[1]));
            timer().startCpuTimer();
            odata[0] = 0;
            for (int i = 1; i < n; i++)
            {
                odata[i] = odata[i] + odata[i - 1];
            }
            timer().endCpuTimer();
        }

        /**
         * CPU stream compaction without using the scan function.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithoutScan(int n, int *odata, const int *idata) {
            // TODO
            timer().startCpuTimer();
            // TODO
            int j = 0;
            for (int i = 0; i < n; i++)
            {
                if (idata[i] != 0)
                {
                    odata[j] = idata[i];
                    ++j;
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
            // TODO
            std::copy(idata, idata + n - 1, &(odata[1]));
            odata[0] = 0;
            map(n, odata);
            for (int i = 1; i < n; i++)
            {
                odata[i] = odata[i] + odata[i - 1];
            }
            int lastNumber = odata[n - 1];
            int* temp = (int*)malloc(n * sizeof(int));
            memcpy(temp, odata, n * sizeof(int));
            if (idata[n - 1] != 0)
            {
                ++lastNumber;
            }

            for (int i = 0; i < n; i++)
            {
                odata[temp[i]] = idata[i];
            }
            free(temp);
            timer().endCpuTimer();
            return lastNumber;
        }
    }
}

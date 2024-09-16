#include <cstdio>
#include "cpu.h"

#include "common.h"

namespace StreamCompaction
{
    namespace CPU
    {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer &timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        struct TimerGuard
        {
            TimerGuard()
            {
                timer().startCpuTimer();
            }
            ~TimerGuard()
            {
                timer().endCpuTimer();
            }
        };

        void scan_untimed(int n, int *odata, const int *idata)
        {
            odata[0] = 0;
            for (int i = 1; i < n; i++)
            {
                odata[i] = odata[i - 1] + idata[i - 1];
            }
        }

        /**
         * CPU scan (prefix sum).
         * For performance analysis, this is supposed to be a simple for loop.
         * (Optional) For better understanding before starting moving to GPU, you can simulate your GPU scan in this function first.
         */
        void scan(int n, int *odata, const int *idata)
        {
            TimerGuard _;
            if (n <= 0)
            {
                return;
            }
            scan_untimed(n, odata, idata);
        }

        /**
         * CPU stream compaction without using the scan function.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithoutScan(int n, int *odata, const int *idata)
        {
            TimerGuard _;
            int count = 0;
            for (int i = 0; i < n; i++)
            {
                int idataElem = idata[i];
                if (idataElem != 0)
                {
                    odata[count++] = idataElem;
                }
            }
            return count;
        }

        /**
         * CPU stream compaction using scan and scatter, like the parallel version.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithScan(int n, int *odata, const int *idata)
        {
            TimerGuard _;
            if (n <= 0)
            {
                return 0;
            }
            int *truthy = static_cast<int *>(malloc(sizeof(int) * n));
            for (int i = 0; i < n; i++)
            {
                truthy[i] = (idata[i] == 0) ? 0 : 1;
            }
            int *indices = static_cast<int *>(malloc(sizeof(int) * n));
            scan_untimed(n, indices, truthy);
            free(truthy);

            int count = 0;
            for (int i = 0; i < n; i++)
            {
                if (idata[i] != 0)
                {
                    odata[indices[i]] = idata[i];
                    count++;
                }
            }
            free(indices);
            return count;
        }
    }
}
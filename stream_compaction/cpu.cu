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
            timer().startCpuTimer();
            // TODO
            int index = 0;
            for (int i = 0; i < n; ++i)
            {
                if (idata[i])
                {
                    odata[index++] = idata[i];
                }
            }
            timer().endCpuTimer();
            return index;
        }

        /**
        * map [1 5 0 1 2 0 3]->[1 1 0 1 1 0 1]
        * data -> bool array
        */
        void compactMap(int n, int* odata, const int* idata)
        {
            for (int i = 0; i < n; ++i)
            {
                odata[i] = 0;
                if (idata[i])
                {
                    odata[i] = 1;
                }
            }
            return;
        }

        /**
         * CPU stream compaction using scan and scatter, like the parallel version.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithScan(int n, int *odata, const int *idata) {
            int* tempdata = new int[n];
            timer().startCpuTimer();
            // TODO
            compactMap(n, tempdata, idata);
            odata[0] = 0;
            for (int i = 1; i < n; ++i)
            {
                odata[i] = odata[i - 1] + tempdata[i - 1];
            }
            int count = odata[n - 1] + (idata[n - 1] ? 1 : 0);
            for (int i = 0; i < n; ++i)
            {
                if (tempdata[i])
                {
                    odata[odata[i]] = idata[i];
                }
            }
            timer().endCpuTimer();
            delete[] tempdata;
            return count;
        }

        void sort(int n, int* odata, const int* idata)
        {
            std::memcpy(odata, idata, n * sizeof(int));
            timer().startCpuTimer();
            std::sort(odata, odata + n);
            timer().endCpuTimer();
        }
    }
}

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
        void scan(int n, int *odata, const int *idata, bool timed) {
            if (timed) timer().startCpuTimer();
            // TODO
            int partialSum = 0;
            for (int i = 0; i < n; ++i) {
                odata[i] = partialSum;
                partialSum += idata[i];
            }
            if (timed) timer().endCpuTimer();
        }

        /**
         * CPU stream compaction without using the scan function.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithoutScan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();
            int numElements = 0;
            for (int i = 0; i < n; ++i) {
                if (idata[i]) odata[numElements++] = idata[i];
            }
            timer().endCpuTimer();
            return numElements;
        }

        int scatter(int n, int* odata, const int* bdata, const int* idata) {
            int numElements = 0;
            for (int i = 0; i < n; ++i) {
                if (bdata[i])  odata[numElements++] = idata[i];
            }
            return numElements;
        }

        /**
         * CPU stream compaction using scan and scatter, like the parallel version.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithScan(int n, int *odata, const int *idata) {
            int* buffer = new int[n];
            timer().startCpuTimer();
            // Create boolean mask
            for (int i = 0; i < n; ++i) {
                buffer[i] = (idata[i] != 0);
            }

            scan(n, odata, idata, 0);

            int numElements = scatter(n, odata, buffer, idata);

            timer().endCpuTimer();
            delete[] buffer;
            return numElements;
        }
    }
}

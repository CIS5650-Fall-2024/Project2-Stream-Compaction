#include <cstdio>
#include "cpu.h"

#include "common.h"

#define GPU_ALGO 0

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
#if GPU_ALGO
#else
            odata[0] = 0; 
            for (int i = 1; i < n; ++i) {
              odata[i] = odata[i - 1] + idata[i - 1]; 
            }
#endif
            timer().endCpuTimer();
        }

        /**
         * CPU stream compaction without using the scan function.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithoutScan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();
            int counter = 0;
            for (int i = 0; i < n; ++i) { 
              if (idata[i] != 0) {
                odata[counter++] = idata[i]; 
              }
            }
            timer().endCpuTimer();
            return counter;
        }

        /**
         * CPU stream compaction using scan and scatter, like the parallel version.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithScan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();

            int* scanArr = new int[n];

            // in odata, 1 if non zero
            for (int i = 0; i < n; ++i) {
              odata[i] = idata[i] == 0 ? 0 : 1; 
            }

            // do scan
            scanArr[0] = 0;
            for (int i = 1; i < n; ++i) {
              scanArr[i] = scanArr[i - 1] + odata[i - 1];
            }

            // scatter
            for (int i = 0; i <= n - 2; ++i) {
              if (scanArr[i] != scanArr[i + 1]) {
                odata[scanArr[i]] = idata[i]; 
              }
            }

            int odataSize = scanArr[n - 1];

            delete[] scanArr; 
            timer().endCpuTimer();
            return odataSize;
        }
    }
}

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

            // make sure elements exist
            if (n <= 0) {
                timer().endCpuTimer();
                return;
            }
            
            // add identity for exclusive scan
            odata[0] = 0;
            for (int i = 1; i < n; i++) {
                // do scan in one big for loop :(
                odata[i] = odata[i - 1] + idata[i - 1];
            }

            timer().endCpuTimer();
        }

        /**
         * CPU stream compaction without using the scan function.
         * @param n number of elements in initial array
         * @param idata input array, not modified
         * @param odata output array
         * @returns the number of elements remaining after compaction.
         */
        int compactWithoutScan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();

            int count = 0;
            // loop over entire array as one big loop, 
            for (int i = 0; i < n; i++) {
                // check if they are zero (throw out) or non-zero (keep)
                if (idata[i] != 0) {
                    odata[count] = idata[i];
                    count++;
                }
            }
            timer().endCpuTimer();
            return count;
        }

        /**
     * CPU stream compaction using scan and scatter, like the parallel version.
     *
     * @returns the number of elements remaining after compaction.
     */
        int compactWithScan(int n, int* odata, const int* idata) {
            timer().startCpuTimer();

            // create temporary array
            int* temp = new int[n];

            // loop over creating boolean array
            for (int i = 0; i < n; ++i) {
                temp[i] = (idata[i] != 0) ? 1 : 0;
            }

            // create array for scan result
            int* scanResult = new int[n];
            scanResult[0] = 0;  
            // loop, exclusive scan
            for (int i = 1; i < n; ++i) {
                scanResult[i] = scanResult[i - 1] + temp[i - 1];
            }

            // final loop, use scan result and boolean result to generate new array
            int count = 0;  
            for (int i = 0; i < n; ++i) {
                if (temp[i] == 1) { 
                    odata[scanResult[i]] = idata[i];
                    count++; 
                }
            }

            // cleanup
            delete[] temp;
            delete[] scanResult;

            timer().endCpuTimer();

            return count;
        }

    }
}

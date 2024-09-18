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
            if (n > 0) {
                odata[0] = idata[0]; // Initialize the first element

                for (int i = 1; i < n; ++i) {
                    odata[i] = odata[i - 1] + idata[i - 1];
                }
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

            int count = 0; // Counter for the number of valid elements

            for (int i = 0; i < n; ++i) {
                if (idata[i] != 0) {
                    odata[count] = idata[i]; // Copy valid element to output array
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
        int compactWithScan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();
            // TODO
            // Create the predicate array
            int* predicate = new int[n];
            for (int i = 0; i < n; i++) {
                predicate[i] = (idata[i] != 0) ? 1 : 0; 
            }

            // Perform an exclusive scan on the predicate array
            int* scanResult = new int[n];
            scanResult[0] = 0; 
            for (int i = 1; i < n; i++) {
                scanResult[i] = scanResult[i - 1] + predicate[i - 1];
            }

            // Scatter valid elements to output array (odata) using the scan result
            int numElementsRemaining = 0;
            for (int i = 0; i < n; i++) {
                if (predicate[i] == 1) {
                    odata[scanResult[i]] = idata[i];
                    numElementsRemaining++;
                }
            }

            timer().endCpuTimer();

            delete[] predicate;
            delete[] scanResult;

            return numElementsRemaining;
        }
    }
}

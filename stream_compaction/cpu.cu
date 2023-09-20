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
            
            // TODO
            odata[0] = 0;
            for (int i = 1; i < n; i++) {
                odata[i] = odata[i - 1] + idata[i - 1];
        
            }



            
        }

        /**
         * CPU stream compaction without using the scan function.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithoutScan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();
            // TODO
            int output = 0;
            for (int i = 0; i < n; i++) {
                if (idata[i] != 0) {
                    odata[output] = idata[i];
                    output += 1;
                }
            }


            timer().endCpuTimer();
            return output;
        }

        /**
         * CPU stream compaction using scan and scatter, like the parallel version.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithScan(int n, int *odata, const int *idata) {
            
            
            // TODO
            
            int* checked = new int[n];
            int* preCheck = new int[n];
            int counter = 0;

            timer().startCpuTimer();

            for (int i = 0; i < n; i++) {
                if (idata[i] != 0) {
                    preCheck[i] = 1;
                    counter += 1;
                }
                else {
                    preCheck[i] = 0;
                }
            }

            scan(n, checked, preCheck);

            for (int i = 0; i < n; i++) {
                if (preCheck[i]==1) {
                    odata[checked[i]] = idata[i];
                }
            
            }
            timer().endCpuTimer();

            delete[] checked;
            delete[] preCheck;
            
            
            return counter;
            
            
        }
    }
}

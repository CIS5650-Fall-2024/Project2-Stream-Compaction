#include <cstdio>
#include "cpu.h"

#include "common.h"
#include <iostream>

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
            if (n == 0) return;

            
            odata[0] = 0;

            
            for (int i = 1; i < n; i++) {
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
            int count = 0;

            for (int i = 0; i < n; i++) {
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
        int compactWithScan(int n, int *odata, const int *idata) {
            int* flag = new int[n];
            int* scannedFlag = new int[n];
            
            
            timer().startCpuTimer();
            // TODO
            for (int i = 0; i < n; i++) {
                flag[i] = (idata[i] != 0)? 1:0;

            }
            


            scannedFlag[0] = 0;


            for (int i = 1; i < n; i++) {
                scannedFlag[i] = scannedFlag[i - 1] + flag[i - 1];
            }
           

            int count = 0;

            for (int i = 0; i < n; i++) {
                if (flag[i] == 1) {
                    if (scannedFlag[i] < n) {
                        odata[scannedFlag[i]] = idata[i];
                    }
                    else {
                        std::cerr << "Error: scannedFlag[" << i << "] out of bounds!" << std::endl;
                    }
                    count++;
                }
            }
            
            timer().endCpuTimer();
            delete[] flag;
            delete[] scannedFlag;
            return count;
        }
    }
}

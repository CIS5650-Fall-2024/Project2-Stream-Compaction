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
            int curr_sum = 0;
            for(int i = 0; i < n; ++i){
                odata[i] = curr_sum;
                curr_sum += idata[i];
            }
            timer().endCpuTimer();
        }

        void scanNoTimer(int n, int *odata, const int *idata) {
            int curr_sum = 0;
            for(int i = 0; i < n; ++i){
                odata[i] = curr_sum;
                curr_sum += idata[i];
            }
        }

        /**
         * CPU stream compaction without using the scan function.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithoutScan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();
            int curr_index = 0;
            for(int i = 0; i < n; ++i){
                if(idata[i] != 0){
                    odata[curr_index] = idata[i];
                    curr_index += 1;
                }
            }
            timer().endCpuTimer();
            return curr_index;
        }

        /**
         * CPU stream compaction using scan and scatter, like the parallel version.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithScan(int n, int *odata, const int *idata) {
            int* boolCheck = new int[n];
            int* intPos = new int[n];

            timer().startCpuTimer();

            for(int i = 0; i < n; ++i){
                boolCheck[i] = (int)(idata[i] != 0);
            }
            scanNoTimer(n, intPos, boolCheck);
            int numPos = intPos[n-1] + boolCheck[n-1]; // get total positives 

            for(int i = 0; i < n; ++i){
                if(boolCheck[i]){
                    odata[intPos[i]] = idata[i];
                }
            }
            timer().endCpuTimer();
            return numPos;
        }
    }
}

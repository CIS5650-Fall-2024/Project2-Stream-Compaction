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
        void scan(int n, int* odata, const int* idata) {
            timer().startCpuTimer();

            int current_sum = 0;
            for (int i = 0; i < n; i++) {
                odata[i] = current_sum;
                current_sum += idata[i];
            }

            timer().endCpuTimer();
        }

        /**
         * CPU stream compaction without using the scan function.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithoutScan(int n, int* odata, const int* idata) {
            timer().startCpuTimer();

            int o_idx = 0;
            for (int i = 0; i < n; i++) {
                if (idata[i] != 0) {
                    odata[o_idx] = idata[i];
                    o_idx++;
                }
            }

            timer().endCpuTimer();
            return o_idx;
        }

        int scatter(const int* idata, int* odata, int* scan_result, int* to_insert, int n) {
            int num_elts = 0;
            for (int i = 0; i < n; i++) {
                if (to_insert[i] == 1) {
                    odata[scan_result[i]] = idata[i];
                    num_elts++;
                }
            }
            return num_elts;
        }

        /**
         * CPU stream compaction using scan and scatter, like the parallel version.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithScan(int n, int* odata, const int* idata) {
            timer().startCpuTimer();

            int* to_insert = new int[n];
            int* scan_result = new int[n];
            
            //scan
            int current_sum = 0;
            for (int i = 0; i < n; i++) {
                scan_result[i] = current_sum;
                current_sum += idata[i];
            }

            //set to_insert to 0 if the element is going to be removed and 1 otherwise
            for (int i = 0; i < n; i++) {
                to_insert[i] = (idata[i] == 0) ? 0 : 1;
            }

            //adjust scan result for proper indices
            for (int i = 1; i < n; i++) {
                scan_result[i] = to_insert[i - 1] + scan_result[i - 1];
            }

            //scatter and get number of elements to return
            int num_elts = scatter(idata, odata, scan_result, to_insert, n);

            delete[] to_insert;
            delete[] scan_result;

            timer().endCpuTimer();
            return num_elts;
        }
    }
}

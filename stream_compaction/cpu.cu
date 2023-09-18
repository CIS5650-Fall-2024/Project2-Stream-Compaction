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

        void scan_impl(int n, int* odata, const int* idata)
        {
            odata[0] = 0;
            for (int i = 1; i < n; i++)
            {
                odata[i] = idata[i - 1] + odata[i - 1];
            }
        }

        /**
         * CPU scan (prefix sum).
         * For performance analysis, this is supposed to be a simple for loop.
         * (Optional) For better understanding before starting moving to GPU, you can simulate your GPU scan in this function first.
         */
        void scan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();
            scan_impl(n, odata, idata);
            timer().endCpuTimer();
        }

        /**
         * CPU stream compaction without using the scan function.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithoutScan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();
            int output_index = 0;
            int curr_data;
            for (int i = 0; i < n; i++)
            {
                curr_data = idata[i];
                if (curr_data != 0)
                {
                    odata[output_index] = curr_data;
                    output_index++;
                }
            }
            timer().endCpuTimer();
            return output_index;
        }

        /**
         * CPU stream compaction using scan and scatter, like the parallel version.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithScan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();
            int *temp = new int[n];
            for (int i = 0; i < n; i++)
            {
                temp[i] = idata[i] != 0 ? 1 : 0;
            }
            int* scan_result = new int[n];
            scan_impl(n, scan_result, temp);

            int final_index = 0;
            for (int i = 0; i < n; i++)
            {
                if (temp[i] == 1)
                {
                    final_index = scan_result[i];
					odata[final_index] = idata[i];
				}
            }
            timer().endCpuTimer();
            return final_index + 1;
        }
    }
}

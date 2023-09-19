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
            int idx = 0;
            for (int i = 0; i < n; i++) {
              if (idata[i] > 0) {
                odata[idx] = idata[i];
                idx++;
              }
            }
            
            timer().endCpuTimer();
            if (idx > 0) return idx;
            return -1;
        }

        /**
         * CPU stream compaction using scan and scatter, like the parallel version.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithScan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();

            // TODO
            std::vector<int> id(n, 0);
            std::vector<int> idx_array(n, 0);

            for (int i = 0; i < n; i++) {
              if (idata[i] > 0) id[i] = 1;
            }

            for (int i = 1; i < n; i++) {
              idx_array[i] = idx_array[i - 1] + id[i - 1];
            }

            int maxIdx = 0;
            for (int i = 0; i < n; i++) {
              if (id[i] > 0) {
                odata[idx_array[i]] = idata[i];
                maxIdx = std::max(idx_array[i], maxIdx);
              }
            }

            
            timer().endCpuTimer();
            if (maxIdx > 0) return maxIdx + 1;
            return -1;
        }

        void sort(int n, int* odata, const int* idata) {
          timer().startCpuTimer();

          std::copy(idata, idata + n, odata);
          std::sort(odata, odata + n);

          timer().endCpuTimer();
        }
    }
}

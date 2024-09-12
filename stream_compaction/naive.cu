#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"

namespace StreamCompaction {
    namespace Naive {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }
        // TODO: __global__
        __global__ void kernNaiveScan(int n, int *odata, const int *idata) {

        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            timer().startGpuTimer();
            int arrSize = n; 
            if (!((n & (n - 1)) == 0)) {  // if n is not a power of 2, pad the array to next power of 2
              arrSize = 1 << ilog2ceil(n); 
            }

            // allocate some arrays on device

            // run kernel

            // copy from device to host

            timer().endGpuTimer();
        }
    }
}

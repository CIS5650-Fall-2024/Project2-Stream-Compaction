#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/scan.h>
#include "common.h"
#include "thrust.h"

namespace StreamCompaction {
    namespace Thrust {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }
        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int* odata, const int* idata) {

            // create thrust device vectors
            thrust::device_vector<int> d_idata(idata, idata + n);
            thrust::device_vector<int> d_odata(n);
            timer().startGpuTimer();

            // use thrust exclusive scan
            thrust::exclusive_scan(d_idata.begin(), d_idata.end(), d_odata.begin());

            timer().endGpuTimer();

            // copy result back
            thrust::copy(d_odata.begin(), d_odata.end(), odata);
        }
    }
}

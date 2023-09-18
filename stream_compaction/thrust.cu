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
        void scan(int n, int *odata, const int *idata) {
            //convert input thrust ptr
            thrust::device_vector<int> thrust_dev_idata = thrust::host_vector<int>(idata, idata+n);

            timer().startGpuTimer();
            // call thrust exclusive scan on data
            thrust::exclusive_scan(thrust_dev_idata.begin(), thrust_dev_idata.end(), thrust_dev_idata.begin());
            timer().endGpuTimer();

            //copy from thrust vec back to host odata
            thrust::copy(thrust_dev_idata.begin(), thrust_dev_idata.end(), odata);
        }
    }
}

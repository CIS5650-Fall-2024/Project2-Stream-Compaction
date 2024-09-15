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
            thrust::host_vector<int> hostData(idata, idata + n);
            thrust::device_vector<int> deviceData = hostData;

            timer().startGpuTimer();
            thrust::exclusive_scan(deviceData.begin(), deviceData.end(), deviceData.begin());
            timer().endGpuTimer();

            memcpy(odata, hostData.data(), n * sizeof(int));
        }
    }
}

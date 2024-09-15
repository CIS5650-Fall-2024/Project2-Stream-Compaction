#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/scan.h>
#include "common.h"
#include "thrust.h"

#define blockSize 512

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
            thrust::device_vector<int> dev_in(idata, idata + n);
            thrust::device_vector<int> dev_out(n, 0);
            cudaDeviceSynchronize();

            nvtxRangePushA("Thrust");
            timer().startGpuTimer();
            thrust::exclusive_scan(dev_in.begin(), dev_in.end(), dev_out.begin());
            timer().endGpuTimer();
            nvtxRangePop();

            thrust::copy(dev_out.begin(), dev_out.end(), odata);
        }
    }
}

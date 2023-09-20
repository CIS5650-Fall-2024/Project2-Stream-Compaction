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
            thrust::host_vector<int> host_idata_vec(idata, idata + n);

            // we can copy a host vector into device simply like this
            thrust::device_vector<int> dev_idata_vec = host_idata_vec;

            timer().startGpuTimer();
            thrust::exclusive_scan(dev_idata_vec.begin(), dev_idata_vec.end(), dev_idata_vec.begin());
            timer().endGpuTimer();

            cudaMemcpy(odata, dev_idata_vec.data().get(), sizeof(int) * n, cudaMemcpyDeviceToHost);
        }
    }
}

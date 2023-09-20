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
            thrust::host_vector<int> host_data(idata, idata + n);
            thrust::device_vector<int> device_data = host_data;            
            timer().startGpuTimer();            
            //thrust::exclusive_scan(idata, idata + n, odata);
            thrust::exclusive_scan(device_data.begin(), device_data.end(), device_data.begin());
            //host_data = device_data;
            timer().endGpuTimer();
            cudaMemcpy(odata, device_data.data().get(), sizeof(int) * n, cudaMemcpyDeviceToHost);            
        }
    }
}

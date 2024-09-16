#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/scan.h>
#include <thrust/sort.h>
#include "common.h"
#include "thrust.h"

namespace StreamCompaction {
    namespace Thrust {
        struct TimerGuard
        {
            TimerGuard()
            {
                timer().startGpuTimer();
            }
            ~TimerGuard()
            {
                timer().endGpuTimer();
            }
        };

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
            size_t numBytes = n * sizeof(int);

            int *dev_idata;
            cudaMalloc(&dev_idata, numBytes);
            cudaMemcpy(dev_idata, idata, numBytes, cudaMemcpyHostToDevice);

            int *dev_odata;
            cudaMalloc(&dev_odata, numBytes);

            {
                TimerGuard _;
                thrust::exclusive_scan(thrust::device, dev_idata, dev_idata + n, dev_odata);
            }

            cudaMemcpy(odata, dev_odata, numBytes, cudaMemcpyDeviceToHost);
            cudaFree(dev_idata);
            cudaFree(dev_odata);
        }
    }
}

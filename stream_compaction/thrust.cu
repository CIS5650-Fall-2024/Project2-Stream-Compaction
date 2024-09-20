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
            int* dev_buffer;
            thrust::device_ptr<int> dev_thrustBuffer;
            cudaMalloc((void**)&dev_buffer, n * sizeof(int));
            checkCUDAError("cudaMalloc dev_buffer failed!");
            dev_thrustBuffer = thrust::device_ptr<int>(dev_buffer);

            cudaMemcpy(dev_buffer, idata, n * sizeof(int), cudaMemcpyHostToDevice);

            timer().startGpuTimer();
            thrust::exclusive_scan(dev_thrustBuffer, dev_thrustBuffer + n, dev_thrustBuffer);
            timer().endGpuTimer();

            cudaMemcpy(odata, dev_buffer, n * sizeof(int), cudaMemcpyDeviceToHost);

            cudaFree(dev_buffer);
        }
    }
}

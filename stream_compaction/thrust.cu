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
            // TODO use `thrust::exclusive_scan`
            // example: for device_vectors dv_in and dv_out:
            // thrust::exclusive_scan(dv_in.begin(), dv_in.end(), dv_out.begin());
            int* dev1, * dev2;
            cudaMalloc((void**)&dev1, n * sizeof(int));
            cudaMalloc((void**)&dev2, n * sizeof(int));
            cudaMemcpy(dev1, idata, sizeof(int) * n, cudaMemcpyHostToDevice);
            thrust::device_ptr<int> dv_in(dev1), dv_out(dev2);
            cudaDeviceSynchronize();
            timer().startGpuTimer();
            thrust::exclusive_scan(dv_in, dv_in + n, dv_out);
            timer().endGpuTimer();
            cudaDeviceSynchronize();
            cudaMemcpy(odata, dev2, sizeof(int) * n, cudaMemcpyDeviceToHost);
            cudaFree(dev1);
            cudaFree(dev2);
        }
    }
}

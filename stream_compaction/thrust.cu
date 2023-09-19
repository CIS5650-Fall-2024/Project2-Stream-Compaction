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
            timer().startGpuTimer();
            // TODO use `thrust::exclusive_scan`
            // example: for device_vectors dv_in and dv_out:
            // thrust::exclusive_scan(dv_in.begin(), dv_in.end(), dv_out.begin());

            int* gpu_odata;
            int* gpu_idata;

            cudaMalloc((void**)&gpu_odata, n * sizeof(int));
            cudaMalloc((void**)&gpu_idata, n * sizeof(int));
            cudaMemcpy(gpu_idata, idata, sizeof(int) * n, cudaMemcpyHostToDevice);

            thrust::device_ptr<int>thrust_dv_in(gpu_idata);
            thrust::device_ptr<int>thrust_dv_out(gpu_odata);

            thrust::exclusive_scan(thrust_dv_in, thrust_dv_in+n, thrust_dv_out);




            cudaMemcpy(odata, gpu_odata, sizeof(int) * n, cudaMemcpyDeviceToHost);
            cudaFree(gpu_odata);
            cudaFree(gpu_idata);


            timer().endGpuTimer();
        }
    }
}

#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/scan.h>
#include <thrust/remove.h>
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
            thrust::device_vector<int> dev_in(idata, idata + n);
            thrust::device_vector<int> dev_out(n);
            timer().startGpuTimer();
            // TODO use `thrust::exclusive_scan`
            // example: for device_vectors dv_in and dv_out:
            // thrust::exclusive_scan(dv_in.begin(), dv_in.end(), dv_out.begin());
            thrust::exclusive_scan(dev_in.begin(), dev_in.end(), dev_out.begin());
            timer().endGpuTimer();
            thrust::copy_n(dev_out.begin(), n, odata);
        }

        struct isZero
        {
            __host__ __device__
                bool operator()(const int x)
            {
                return (x == 0);
            }
        };

        int compact(int n, int* odata, const int* idata) {
            thrust::device_vector<int> dev_in(idata, idata + n);
            timer().startGpuTimer();
            // TODO
            thrust::detail::normal_iterator<thrust::device_ptr<int>> endIter = thrust::remove_if(dev_in.begin(), dev_in.end(), isZero());
            timer().endGpuTimer();
            thrust::copy(dev_in.begin(), endIter, odata);
            return endIter - dev_in.begin();
        }
    }
}

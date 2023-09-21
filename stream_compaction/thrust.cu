#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/scan.h>
#include "common.h"
#include "thrust.h"
#include <thrust/sort.h>

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
            int* dev_read;
            int* dev_write;

            cudaMalloc((void**)&dev_read, n * sizeof(int));
            checkCUDAError("cudaMalloc dev_read failed!");

            cudaMalloc((void**)&dev_write, n * sizeof(int));
            checkCUDAError("cudaMalloc dev_write failed!");

            // copy data from CPU to GPU
            cudaMemcpy(dev_read, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            checkCUDAError("Memcpy idata(host) to dev_read(device) failed!");

            thrust::device_ptr<int> thrust_dev_read(dev_read);
            thrust::device_ptr<int> thrust_dev_write(dev_write);

            timer().startGpuTimer();
            // TODO use `thrust::exclusive_scan`
            // example: for device_vectors dv_in and dv_out:
            // thrust::exclusive_scan(dv_in.begin(), dv_in.end(), dv_out.begin());
            thrust::exclusive_scan(thrust_dev_read, thrust_dev_read + n, dev_write);
            timer().endGpuTimer();

            cudaMemcpy(odata, dev_write, n * sizeof(int), cudaMemcpyDeviceToHost);
            checkCUDAError("Memcpy dev_write(device) to odata(host) failed!");

            cudaFree(dev_read);
            cudaFree(dev_write);
        }

        void sort(int n, int* odata, const int* idata) 
        {
            thrust::device_vector<int> thrust_dev_data(idata, idata + n);
            
            timer().startGpuTimer();
            thrust::sort(thrust_dev_data.begin(), thrust_dev_data.end());
            timer().endGpuTimer();

            cudaMemcpy(odata, thrust_dev_data.data().get(), n * sizeof(int), cudaMemcpyDeviceToHost);
        }
    }
}

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
            
            // create the input buffer
            thrust::device_vector<int> input_buffer (idata, idata + n);

            // create the output buffer
            thrust::device_vector<int> output_buffer (n);

            // start the timer
            timer().startGpuTimer();
            
            // use thrust::exclusive_scan
            thrust::exclusive_scan(
                input_buffer.begin(),
                input_buffer.end(),
                output_buffer.begin()
            );

            // stop the timer
            timer().endGpuTimer();

            // copy the data in the output buffer to the actual output memory
            thrust::copy(
                output_buffer.begin(),
                output_buffer.end(),
                odata
            );

            // avoid calling the original end timer function afterwards by returning
            return;

            timer().endGpuTimer();
        }
    }
}

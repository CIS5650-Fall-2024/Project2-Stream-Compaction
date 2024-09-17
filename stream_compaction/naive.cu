#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"

namespace StreamCompaction {
    namespace Naive {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }
        
        __global__ void kernNaiveScan(int n, int d, int* odata, const int* idata)
        {
            int index = (blockIdx.x * blockDim.x) + threadIdx.x;
            if (index >= n) return;

            int pow2 = (1 << (d - 1));
            if (index >= pow2)
            {
                // Combine
                odata[index] = idata[index - pow2] + idata[index];
            }
            else
            {
                // Unused this iteration - Copy over to the other array
                odata[index] = idata[index];
            }
        }

        __global__ void kernShiftRight(int n, int s, int* odata, const int* idata)
        {
            int index = (blockIdx.x * blockDim.x) + threadIdx.x;
            if (index >= n) return;

            // Shift to the right
            // Fill newly empty slots with 0s
            odata[index] = index >= s ? idata[index - s] : 0;
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            // Allocate device arrays
            int *dev_odata, *dev_idata;

            cudaMalloc((void**)&dev_odata, n * sizeof(int));
            checkCUDAError("cudaMalloc Naive::scan::dev_odata failed!");

            cudaMalloc((void**)&dev_idata, n * sizeof(int));
            checkCUDAError("cudaMalloc Naive::scan::dev_idata failed!");

            cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);

            dim3 gridDim((n + blockSize - 1) / blockSize);

            // Number of levels needed for the naive parallel scan
            int depth_max = ilog2ceil(n);

            timer().startGpuTimer();

            // Perform an Inclusive Scan
            for (int d = 1; d <= depth_max; ++d)
            {
                kernNaiveScan<<<gridDim, blockSize>>>(n, d, dev_odata, dev_idata);

                int *tmp = dev_idata;
                dev_idata = dev_odata;
                dev_odata = tmp;
            }

            // Inclusive Scan -> Exclusive Scan conversion
            kernShiftRight<<<gridDim, blockSize>>>(n, 1, dev_odata, dev_idata);

            timer().endGpuTimer();

            // Copy the output data
            cudaMemcpy(odata, dev_odata, n * sizeof(int), cudaMemcpyDeviceToHost);

            // Free device arrays
            cudaFree(dev_odata);
            cudaFree(dev_idata);
            checkCUDAError("cudaFree Naive::scan failed!");
        }
    }
}

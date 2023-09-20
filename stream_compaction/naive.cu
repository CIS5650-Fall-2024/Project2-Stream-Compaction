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
        // TODO: __global__
        __global__ void kernNaiveScan(int n, int d, int *odata, const int *idata) {
            int k = (blockIdx.x * blockDim.x) + threadIdx.x;
            if (k >= n) return;

            int pow2_dm1 = 1 << (d - 1);
            if (k >= pow2_dm1) {
                odata[k] = idata[k - pow2_dm1] + idata[k];
            } else {
                odata[k] = idata[k];
            }
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            int* dev_odata;
            cudaMalloc((void**)&dev_odata, n * sizeof(int));
            checkCUDAError("cudaMalloc dev_odata failed!");
            int* dev_idata;
            cudaMalloc((void**)&dev_idata, n * sizeof(int));
            cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            checkCUDAError("cudaMalloc and cudaMemcpy dev_idata failed!");
            cudaDeviceSynchronize();
            timer().startGpuTimer();
            // TODO
            int blockSize = 128;
            dim3 threadsPerBlock(blockSize);
            dim3 fullBlocksPerGrid((n + blockSize - 1) / blockSize);
            for (int d = 1; d <= ilog2ceil(n); d++) {
                kernNaiveScan<<<fullBlocksPerGrid, threadsPerBlock>>>(n, d, dev_odata, dev_idata);
                std::swap(dev_odata, dev_idata);
            }
            timer().endGpuTimer();
            // dev_idata (not dev_odata) now contains an inclusive scan
            cudaMemcpy(odata + 1, dev_idata, (n - 1) * sizeof(int), cudaMemcpyDeviceToHost);
            checkCUDAError("cudaMemcpy dev_odata failed!");
            odata[0] = 0;
            cudaFree(dev_idata);
            cudaFree(dev_odata);
        }
    }
}

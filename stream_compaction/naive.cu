#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"

#define blockSize 128

namespace StreamCompaction {
    namespace Naive {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }
        // TODO: __global__
        __global__ void kernScan(int n, int depth, int* odata, int* idata) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= n) return;

            if (idx < depth) {
                odata[idx] = idata[idx];
                return;
            }

            odata[idx] = idata[idx - depth] + idata[idx];
            return;
        }

        __global__ void kernToExclusive(int n, int* odata, int* idata) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= n) return;

            if (idx == 0) {
                odata[idx] = 0;
            }
            else {
                odata[idx] = idata[idx - 1];
            }
            return;
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            int* dev_odata;
            int* dev_idata;

            cudaMalloc((void**)&dev_odata, n * sizeof(int));
            checkCUDAError("cudaMalloc dev_odata failed");
            cudaMalloc((void**)&dev_idata, n * sizeof(int));
            checkCUDAError("cudaMalloc dev_idata failed");

            cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            checkCUDAError("cudaMemcpy to dev_idata failed");

            timer().startGpuTimer();
            // TODO
            dim3 blocksPerGrid((n + blockSize - 1) / blockSize);

            for (int d = 1; d <= ilog2ceil(n); ++d) {
                kernScan<<<blocksPerGrid, blockSize>>>(n, 1 << (d - 1), dev_odata, dev_idata);
                std::swap(dev_odata, dev_idata);
            }
            
            // above result is inclusive, need to convert to exclusive scan
            // because of the sawp, latest result is in dev_idata
            kernToExclusive<<<blocksPerGrid, blockSize>>>(n, dev_odata, dev_idata);

            timer().endGpuTimer();

            cudaMemcpy(odata, dev_odata, n * sizeof(int), cudaMemcpyDeviceToHost);
            checkCUDAError("cudaMemcpy from dev_odata failed");

            cudaFree(dev_odata);
            checkCUDAError("cudaFree dev_odata failed");
            cudaFree(dev_idata);
            checkCUDAError("cudaFree dev_idata failed");
        }
    }
}

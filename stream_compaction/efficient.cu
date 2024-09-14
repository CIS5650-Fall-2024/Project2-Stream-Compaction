#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

#define blockSize 128

namespace StreamCompaction {
    namespace Efficient {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        __global__ void kernUpSweep(int n, int d, int* idata) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            int k = (1 << (d + 1)) * idx;
            if (k >= n) return;

            idata[k + 1 << (d + 1) - 1] += idata[k + 1 << d - 1];
        }

        __global__ void kernDownSweep(int n, int d, int* idata) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            int k = (1 << (d + 1)) * idx;
            if (k >= n) return;

            int t = idata[k + 1 << d - 1];
            idata[k + 1 << d - 1] = idata[k + 1 << (d + 1) - 1];
            idata[k + 1 << (d + 1) - 1] += t;
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
        void scan(int n, int *odata, const int *idata, bool time) {
            int log2CeilN = ilog2ceil(n);
            int n_padded = 1 << log2CeilN;

            int* dev_odata;
            int* dev_idata;

            cudaMalloc((void**)&dev_odata, n_padded * sizeof(int));
            checkCUDAError("cudaMalloc dev_odata failed");
            cudaMalloc((void**)&dev_idata, n_padded * sizeof(int));
            checkCUDAError("cudaMalloc dev_idata failed");

            // pad to zero for entries >= n
            cudaMemset(dev_odata, 0, n_padded * sizeof(int));
            checkCUDAError("cudaMemset dev_odata failed");
            cudaMemset(dev_idata, 0, n_padded * sizeof(int));
            checkCUDAError("cudaMemset dev_idata failed");

            cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            checkCUDAError("cudaMemcpy to dev_idata failed");

            if (time)
                timer().startGpuTimer();
            // TODO

            // upsweep
            for (int d = 0; d < log2CeilN; d++) {
                // "for all k = 0 to n-1 by 2^(d+1) in parallel"
                dim3 blocksPerGrid((n_padded / (1 << (d + 1)) + blockSize - 1) / blockSize);
                kernUpSweep<<<blocksPerGrid, blockSize>>>(n, d, dev_idata);
            }

            // set root to zero
            cudaMemset(dev_idata + n_padded - 1, 0, sizeof(int));

            // downseep
            for (int d = log2CeilN - 1; d >= 0; d--) {
                // "for all k = 0 to n-1 by 2^(d+1) in parallel"
                dim3 blocksPerGrid((n_padded / (1 << (d + 1)) + blockSize - 1) / blockSize);
                kernDownSweep<<<blocksPerGrid, blockSize>>>(n, d, dev_idata);
            }

            //dim3 blocksPerGrid((n + blockSize - 1) / blockSize);
            //kernToExclusive<<<blocksPerGrid, blockSize>>>(n, dev_odata, dev_idata);

            if (time)
                timer().endGpuTimer();

            cudaMemcpy(odata, dev_odata, n * sizeof(int), cudaMemcpyDeviceToHost);
            checkCUDAError("cudaMemcpy from dev_odata failed");

            cudaFree(dev_odata);
            checkCUDAError("cudaFree dev_odata failed");
            cudaFree(dev_idata);
            checkCUDAError("cudaFree dev_idata failed");
        }

        /**
         * Performs stream compaction on idata, storing the result into odata.
         * All zeroes are discarded.
         *
         * @param n      The number of elements in idata.
         * @param odata  The array into which to store elements.
         * @param idata  The array of elements to compact.
         * @returns      The number of elements remaining after compaction.
         */
        int compact(int n, int *odata, const int *idata) {
            timer().startGpuTimer();
            // TODO
            timer().endGpuTimer();
            return -1;
        }
    }
}

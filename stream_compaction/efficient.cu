#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>
#include "common.h"
#include "efficient.h"

__device__ inline int twoPow(int d) {
    return (1 << (d));
}

inline int twoPow_Host(int d) {
    return (1 << (d));
}

namespace StreamCompaction {
    namespace Efficient {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        __global__ void upSweep(int n, int d, int* x) {
            int idx = blockDim.x * blockIdx.x + threadIdx.x;
            if (idx >= n) return;
            if (idx % twoPow(d + 1) != 0) return;
            x[idx + twoPow(d + 1) - 1] += x[idx + (twoPow(d)) - 1];
        }

        __global__ void downSweep(int n, int d, int* x) {
            int idx = blockDim.x * blockIdx.x + threadIdx.x;
            if (idx >= n) return;
            if (idx % twoPow(d + 1) != 0) return;

            int tmp = x[idx + twoPow(d) - 1];
            x[idx + twoPow(d) - 1] = x[idx + twoPow(d + 1) - 1];
            x[idx + twoPow(d + 1) - 1] += tmp;
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            
            // TODO
            int size = twoPow_Host(ilog2ceil(n));
            dim3 blockPerGrids((size + blockSize - 1) / blockSize);
            int* dev_idata;
            cudaMalloc((void**)&dev_idata, size * sizeof(int));
            cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            cudaDeviceSynchronize();

            timer().startGpuTimer();
            // UpSweep
            for (int d = 0; d < ilog2ceil(size); d++) {
                upSweep << <blockPerGrids, blockSize >> > (n, d, dev_idata);
                cudaDeviceSynchronize();
            }
            cudaMemset(dev_idata + size - 1, 0, sizeof(int));

            // DownSweep
            for (int d = ilog2ceil(size) - 1; d >= 0; d--) {
                downSweep << <blockPerGrids, blockSize >> > (n, d, dev_idata);
                cudaDeviceSynchronize();
            }
            timer().endGpuTimer();

            cudaMemcpy(odata, dev_idata, n * sizeof(int), cudaMemcpyDeviceToHost);

            cudaFree(dev_idata);
            
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
            
            int* dev_bools;
            int* dev_indices;
            int* dev_idata;
            int* dev_odata;
            int size = twoPow_Host(ilog2ceil(n));
            int cnt = 0;

            dim3 blockPerGrids((n + blockSize - 1) / blockSize);
            dim3 fullBlockPerGrids((size + blockSize - 1) / blockSize);

            cudaMalloc((void**)&dev_bools, size * sizeof(int));
            cudaMalloc((void**)&dev_indices, size * sizeof(int));
            cudaMalloc((void**)&dev_idata, size * sizeof(int));
            cudaMalloc((void**)&dev_odata, size * sizeof(int));
            cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);

            timer().startGpuTimer();
            // TODO
            Common::kernMapToBoolean << <blockPerGrids, blockSize >> > (n, dev_bools, dev_idata);
            cudaDeviceSynchronize();
            cudaMemcpy(dev_indices, dev_bools, n * sizeof(int), cudaMemcpyDeviceToDevice);

            // scan
            for (int d = 0; d < ilog2ceil(size); d++) {
                upSweep << <fullBlockPerGrids, blockSize >> > (n, d, dev_indices);
            }
            cudaMemset(dev_indices + size - 1, 0, sizeof(int));

            for (int d = ilog2ceil(size) - 1; d >= 0; d--) {
                downSweep << <fullBlockPerGrids, blockSize >> > (n, d, dev_indices);
            }
            
            Common::kernScatter << <blockPerGrids, blockSize >> > (n, dev_odata, dev_idata, dev_bools, dev_indices);
            timer().endGpuTimer();
            cudaMemcpy(&cnt, dev_indices + size - 1, sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(odata, dev_odata, cnt * sizeof(int), cudaMemcpyDeviceToHost);

            cudaFree(dev_idata);
            cudaFree(dev_odata);
            cudaFree(dev_indices);
            cudaFree(dev_bools);
            

            return cnt;
        }
    }
}

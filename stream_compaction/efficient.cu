#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

#define logBlockSize 6

namespace StreamCompaction {
    namespace Efficient {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        __global__ void kernUp(int d, int n, int* odata) {
            const int p = 1 << d;
            const int k = (blockIdx.x * blockDim.x + threadIdx.x) * 2 * p;
            if (k < n) {
                odata[2 * p + k - 1] += odata[p + k - 1];
            }
            odata[n - 1] = 0;
        }

        __global__ void kernDown(int d, int n, int* odata) {
            const int p = 1 << d;
            const int k = (blockIdx.x * blockDim.x + threadIdx.x) * 2 * p;
            if (k < n) {
                int t = odata[p + k - 1];
                odata[p + k - 1] = odata[2 * p + k - 1];
                odata[2 * p + k - 1] += t;
            }
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            const int blockSize = 1 << logBlockSize;
            int* dev_data;

            // allocate enough memory to the next power of two 
            const int log = ilog2ceil(n);
            const int expN = 1 << log;
            const int extra = expN - n;

            cudaMalloc((void**)&dev_data, expN * sizeof(int));

            cudaMemcpy(dev_data + extra, idata, n * sizeof(int), cudaMemcpyHostToDevice);

            timer().startGpuTimer();

            // up
            for (int d = 0; d < log; d++) {
                kernUp << <dim3(max(1, expN >> (d + logBlockSize + 1))), blockSize >> > (d, expN, dev_data);
            }

            // down
            for (int d = log - 1; d >= 0; d--) {
                kernDown << <dim3(max(1, expN >> (d + logBlockSize + 1))), blockSize >> > (d, expN, dev_data);
            }

            timer().endGpuTimer();

            cudaMemcpy(odata, dev_data + extra, n * sizeof(int), cudaMemcpyDeviceToHost);
            cudaFree(dev_data);
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
            const int blockSize = 1 << logBlockSize;

            int* dev_idata;
            int* dev_hasValue;
            int* dev_scanned;
            int* dev_odata;

            int total = 0;

            const int log = ilog2ceil(n);
            const int expN = 1 << log;
            const int extra = expN - n;

            cudaMalloc((void**)&dev_idata, n * sizeof(int));
            cudaMalloc((void**)&dev_hasValue, n * sizeof(int));

            cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            cudaMemcpy(dev_hasValue, idata, n * sizeof(int), cudaMemcpyHostToDevice);

            const dim3 fullBlocksPerGrid((expN + blockSize - 1) / blockSize);

            timer().startGpuTimer();

            // get 0 and non 0 elements
            StreamCompaction::Common::kernMapToBoolean <<<fullBlocksPerGrid, blockSize >>>(n, dev_hasValue, dev_idata);

            // scan

            cudaMalloc((void**)&dev_scanned, expN * sizeof(int));

            cudaMemcpy(dev_scanned + extra, dev_hasValue, n * sizeof(int), cudaMemcpyDeviceToDevice);

            for (int d = 0; d < log; d++) {
                kernUp << <dim3(max(1, expN >> (d + logBlockSize + 1))), blockSize >> > (d, expN, dev_scanned);
            }

            for (int d = log - 1; d >= 0; d--) {
                kernDown << <dim3(max(1, expN >> (d + logBlockSize + 1))), blockSize >> > (d, expN, dev_scanned);
            }

            cudaMemcpy(&total, dev_scanned + expN - 1, sizeof(int), cudaMemcpyDeviceToHost);
            if (idata[n - 1] > 0) {
                total++;
            }

            // scatter
            cudaMalloc((void**)&dev_odata, expN * sizeof(int));

            StreamCompaction::Common::kernScatter << <fullBlocksPerGrid, blockSize >> > (n, dev_odata, dev_idata, dev_hasValue, dev_scanned + extra);

            timer().endGpuTimer();

            // cudaMemcpy(odata, dev_scanned + extra, n * sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(odata, dev_odata, expN * sizeof(int), cudaMemcpyDeviceToHost);

            cudaFree(dev_idata);
            cudaFree(dev_hasValue);
            cudaFree(dev_scanned);
            cudaFree(dev_odata);

            
            return total;
        }
    }
}

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
            timer().startGpuTimer();
            int* dev_data;

            // allocate enough memory to the next power of two 
            const int expN = 1 << ilog2ceil(n);
            const int extra = expN - n;

            cudaMalloc((void**)&dev_data, expN * sizeof(int));

            cudaMemcpy(dev_data + extra, idata, n * sizeof(int), cudaMemcpyHostToDevice);

            // up
            for (int d = 0; d < ilog2ceil(n); d++) {
                kernUp << <dim3(1 + (expN/ (1 << (d + 1)))), blockSize >> > (d, expN, dev_data);
            }

            // down
            for (int d = ilog2ceil(n) - 1; d >= 0; d--) {
                kernDown << <dim3(1 + (expN / (1 << (d + 1)))), blockSize >> > (d, expN, dev_data);
            }

            cudaMemcpy(odata, dev_data + extra + 1, (n - 1) * sizeof(int), cudaMemcpyDeviceToHost);
            odata[n - 1] += idata[n - 1];
            if (n > 1) {
                odata[n - 1] += odata[n - 2];
            }

            cudaFree(dev_data);

            timer().endGpuTimer();
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
            int* dev_idata;
            int* dev_hasValue;
            int* dev_scanned;
            int* dev_odata;

            int total = 0;

            const int expN = 1 << ilog2ceil(n);
            const int extra = expN - n;

            cudaMalloc((void**)&dev_idata, n * sizeof(int));
            cudaMalloc((void**)&dev_hasValue, n * sizeof(int));

            cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            cudaMemcpy(dev_hasValue, idata, n * sizeof(int), cudaMemcpyHostToDevice);

            const dim3 fullBlocksPerGrid((expN + blockSize - 1) / blockSize);

            // get 0 and non 0 elements
            StreamCompaction::Common::kernMapToBoolean <<<fullBlocksPerGrid, blockSize >>>(n, dev_hasValue, dev_idata);

            // scan

            cudaMalloc((void**)&dev_scanned, expN * sizeof(int));

            cudaMemcpy(dev_scanned + extra, dev_hasValue, n * sizeof(int), cudaMemcpyDeviceToDevice);

            for (int d = 0; d < ilog2ceil(n); d++) {
                kernUp << <dim3(1 + (expN / (1 << (d + 1)))), blockSize >> > (d, expN, dev_scanned);
            }

            for (int d = ilog2ceil(n) - 1; d >= 0; d--) {
                kernDown << <dim3(1 + (expN / (1 << (d + 1)))), blockSize >> > (d, expN, dev_scanned);
            }

            cudaMemcpy(&total, dev_scanned + expN - 1, sizeof(int), cudaMemcpyDeviceToHost);
            if (idata[n - 1] > 0) {
                total++;
            }

            // scatter
            cudaMalloc((void**)&dev_odata, expN * sizeof(int));

            StreamCompaction::Common::kernScatter << <fullBlocksPerGrid, blockSize >> > (n, dev_odata, dev_idata, dev_hasValue, dev_scanned + extra);

            // cudaMemcpy(odata, dev_scanned + extra, n * sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(odata, dev_odata, expN * sizeof(int), cudaMemcpyDeviceToHost);

            cudaFree(dev_idata);
            cudaFree(dev_hasValue);
            cudaFree(dev_scanned);
            cudaFree(dev_odata);

            timer().endGpuTimer();
            return total;
        }
    }
}

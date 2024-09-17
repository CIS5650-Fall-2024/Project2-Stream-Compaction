#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

#define blockSize 512

namespace StreamCompaction {
    namespace Efficient {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        __global__ void kernUpSweep(int n, int* odata, int offset)
        {
            int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
            if (idx >= n) return;

            int index = ((idx + 1) << offset) - 1;
            offset = 1 << (offset - 1);
            odata[index] += odata[index - offset];
        }

        __global__ void kernDownSweep(int n, int* odata, int offset)
        {
            int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
            if (idx >= n) return;

            int index = ((idx + 1) << offset) - 1;
            offset = 1 << (offset - 1);

            int t = odata[index - offset];
            odata[index - offset] = odata[index];
            odata[index] += t;
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            // round size to power of 2
            int d = ilog2ceil(n);
            int blockNum = 0;
            int paddedNum = 1 << d;

            int* dev_obuffer;
            cudaMalloc((void**)&dev_obuffer, paddedNum * sizeof(int));
            cudaMemset(dev_obuffer + n, 0, (paddedNum - n) * sizeof(int));
            cudaMemcpy(dev_obuffer, idata, sizeof(int) * n, cudaMemcpyHostToDevice);

            nvtxRangePushA("Efficient");
            timer().startGpuTimer();

            int taskNum = paddedNum;
            for (int i = 1; i <= d; ++i)
            {
                taskNum = taskNum >> 1;
                blockNum = (taskNum + blockSize - 1) / blockSize;
                kernUpSweep << < blockNum, blockSize >> > (taskNum, dev_obuffer, i);
            }

            cudaMemset(dev_obuffer + paddedNum - 1, 0, sizeof(int));
            for (int i = d; i >= 1; --i)
            {
                blockNum = (taskNum + blockSize - 1) / blockSize;
                kernDownSweep << < blockNum, blockSize >> > (taskNum, dev_obuffer, i);
                taskNum = taskNum << 1;
            }

            timer().endGpuTimer();
            nvtxRangePop();

            cudaMemcpy(odata, dev_obuffer, sizeof(int) * n, cudaMemcpyDeviceToHost);
            cudaFree(dev_obuffer);
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

            int d = ilog2ceil(n);
            int blockNum = 0;
            int paddedNum = 1 << d;

            int* dev_idata;
            cudaMalloc((void**)&dev_idata, paddedNum * sizeof(int));
            cudaMemset(dev_idata, 0, paddedNum * sizeof(int));
            cudaMemcpy(dev_idata, idata, sizeof(int) * n, cudaMemcpyHostToDevice);

            int* dev_bools;
            cudaMalloc((void**)&dev_bools, paddedNum * sizeof(int));
            cudaMemset(dev_bools, 0, paddedNum * sizeof(int));

            int* dev_odata;
            cudaMalloc((void**)&dev_odata, paddedNum * sizeof(int));
            cudaMemset(dev_odata, 0, paddedNum * sizeof(int));

            timer().startGpuTimer();
            
            // map to bool
            blockNum = (paddedNum + blockSize - 1) / blockSize;
            Common::kernMapToBoolean <<< blockNum, blockSize >>> (paddedNum, dev_bools, dev_idata);

            int taskNum = paddedNum;
            for (int i = 1; i <= d; ++i)
            {
                taskNum = taskNum >> 1;
                blockNum = (taskNum + blockSize - 1) / blockSize;
                kernUpSweep << < blockNum, blockSize >> > (taskNum, dev_bools, i);
            }

            cudaMemset(dev_bools + paddedNum - 1, 0, sizeof(int));
            for (int i = d; i >= 1; --i)
            {
                blockNum = (taskNum + blockSize - 1) / blockSize;
                kernDownSweep << < blockNum, blockSize >> > (taskNum, dev_bools, i);
                taskNum = taskNum << 1;
            }

            blockNum = (paddedNum + blockSize - 1) / blockSize;
            Common::kernScatter << < blockNum, blockSize >> > (paddedNum, dev_odata, dev_idata, nullptr, dev_bools);

            // copy len back to host
            int len = 0;
            cudaMemcpy(&len, dev_bools + paddedNum - 1, sizeof(int), cudaMemcpyDeviceToHost);

            timer().endGpuTimer();

            cudaMemcpy(odata, dev_odata, sizeof(int) * len, cudaMemcpyDeviceToHost);
            cudaFree(dev_odata);
            cudaFree(dev_idata);
            cudaFree(dev_bools);


            return len;
        }
    }
}

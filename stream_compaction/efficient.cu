#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

namespace StreamCompaction {
    namespace Efficient {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        __global__ void kernUpSweep(int n, int d, int* odata) {
            int index = (blockIdx.x * blockDim.x + threadIdx.x) * (1 << (d + 1));

            if (index < n) {
                odata[index + (1 << (d + 1)) - 1] += odata[index + (1 << d) - 1];
            }
        }

        __global__ void kernDownSweep(int n, int d, int *odata) {
            int index = (threadIdx.x + blockDim.x * blockIdx.x) + (1 << (d + 1));

            if (index < n) {
                // preserve the left child value
                int temp = odata[index + (1 << d) - 1];
                // left child copies the parent value
                odata[index + (1 << d) - 1] = odata[index + (1 << (d + 1)) - 1];
                // right child addes the parent value and the preserved left child value
                odata[index + (1 << (d + 1)) - 1] = temp + odata[index + (1 << (d + 1)) - 1];
            }
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            int* dev_out;

            const int blockSize = 128;

            cudaMalloc((void**)&dev_out, n * sizeof(int));

            cudaMemcpy(dev_out, idata, n * sizeof(int), cudaMemcpyHostToDevice);

            timer().startGpuTimer();
            // TODO
            // up sweep 
            for (int d = 0; d <= ilog2ceil(n) - 1; ++d) {
                int gridSize = (n / (1 << (d + 1)) + blockSize - 1) / blockSize;
                if (gridSize < 1) gridSize = 1;

                kernUpSweep << <gridSize, blockSize >> > (n, d, dev_out);
                checkCUDAErrorFn("up sweep failed!");
            }

            /*int* upSweepData = new int[n];
            cudaMemcpy(upSweepData, dev_out, n * sizeof(int), cudaMemcpyDeviceToHost);

            for (int i = 0; i < n; ++i) {
                printf("%d ", upSweepData[i]);
            }
            */

            // make the last value to 0
            int zero = 0;
            cudaMemcpy(&dev_out[n - 1], &zero, sizeof(int), cudaMemcpyHostToDevice);

            // down sweep
            for (int d = ilog2ceil(n) - 1; d >= 0; --d) {
                int gridSize = (n / (1 << (d + 1)) + blockSize - 1) / blockSize;
                if (gridSize < 1) gridSize = 1;

                kernDownSweep << <gridSize, blockSize >> > (n, d, dev_out);
                checkCUDAErrorFn("down sweep failed");
            }

            timer().endGpuTimer();

            cudaMemcpy(odata, dev_out, n * sizeof(int), cudaMemcpyDeviceToHost);

            // free memory
            cudaFree(dev_out);

            cudaDeviceSynchronize();
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

#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

#define blockSize 128
#define EFFICIENT 1

namespace StreamCompaction {
    namespace Efficient {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        __global__ void kernUpSweep(int n, int d, int* odata) {
#if EFFICIENT
            int index = (blockIdx.x * blockDim.x + threadIdx.x) * (2 << d);
#else
            int index = (blockIdx.x * blockDim.x + threadIdx.x) * (1 << (d + 1));
#endif

            if (index < n) {
                odata[index + (1 << (d + 1)) - 1] += odata[index + (1 << d) - 1];
            }
        }


        __global__ void kernSetLastElementToZero(int n, int* odata) {
            odata[n - 1] = 0;
        }

        __global__ void kernDownSweep(int n, int d, int* odata) {
#if EFFICIENT
            int index = (blockIdx.x * blockDim.x + threadIdx.x) * (2 << d);
#else
            int index = (blockIdx.x * blockDim.x + threadIdx.x) * (1 << (d + 1));
#endif

            if (index < n) {
                // preserve the left child value
                int temp = odata[index + (1 << d) - 1];
                // left child copies the parent value
                odata[index + (1 << d) - 1] = odata[index + (1 << (d + 1)) - 1];
                // right child addes the parent value and the preserved left child value
                odata[index + (1 << (d + 1)) - 1] += temp;
            }
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            int* dev_out;

            const int log2ceil = ilog2ceil(n);
            const int fullSize = 1 << log2ceil;

            cudaMalloc((void**)&dev_out, fullSize * sizeof(int));
            cudaMemset(dev_out, 0, fullSize * sizeof(int));
            cudaMemcpy(dev_out, idata, n * sizeof(int), cudaMemcpyHostToDevice);

            timer().startGpuTimer();
            // up sweep 
            for (int d = 0; d <= log2ceil - 1; ++d) {
#if EFFICIENT   
                // Adjust the grid size based on the depth of the sweep
                dim3 gridSize = (fullSize / (2 << d) + blockSize - 1) / blockSize;
#else
                dim3 gridSize = (fullSize / (2 << d) + blockSize - 1) / blockSize;
#endif

                kernUpSweep << <gridSize, blockSize >> > (fullSize, d, dev_out);
                checkCUDAErrorFn("up sweep failed!");
            }

            // set the last value to 0
            kernSetLastElementToZero << <1, 1 >> > (fullSize, dev_out);
            checkCUDAErrorFn("set last element to zero failed!");

            // down sweep
            for (int d = log2ceil - 1; d >= 0; --d) {
#if EFFICIENT   
                // Adjust the grid size based on the depth of the sweep
                dim3 gridSize = (fullSize / (2 << d) + blockSize - 1) / blockSize;
#else
                dim3 gridSize = (fullSize / (2 << d) + blockSize - 1) / blockSize;
#endif

                kernDownSweep << <gridSize, blockSize >> > (fullSize, d, dev_out);
                checkCUDAErrorFn("down sweep failed");
            }

            timer().endGpuTimer();

            cudaMemcpy(odata, dev_out, n * sizeof(int), cudaMemcpyDeviceToHost);

            // free memory
            cudaFree(dev_out);
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
            int* dev_in;
            int* dev_out;

            int* dev_bools;
            int* dev_scan;

            int boolLastVal, scanLastVal;

            const int log2ceil = ilog2ceil(n);
            const int fullSize = 1 << log2ceil;

            dim3 gridSize = (n + blockSize - 1) / blockSize;

            cudaMalloc((void**)&dev_bools, n * sizeof(int));
            checkCUDAErrorFn("malloc dev_bools failed!");

            cudaMalloc((void**)&dev_scan, fullSize * sizeof(int));
            cudaMemset(dev_scan, 0, fullSize * sizeof(int));
            checkCUDAErrorFn("malloc dev_scan failed!");

            cudaMalloc((void**)&dev_in, n * sizeof(int));
            checkCUDAErrorFn("malloc dev_in failed!");
            cudaMemcpy(dev_in, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            checkCUDAErrorFn("copy idata to dev_in failed!");
            
            cudaMalloc((void**)&dev_out, n * sizeof(int));
            checkCUDAErrorFn("malloc dev_out failed!");

            timer().startGpuTimer();
            // map the bool array
            StreamCompaction::Common::kernMapToBoolean << <gridSize, blockSize >> > (n, dev_bools, dev_in);
            checkCUDAErrorFn("map bool array failed!");

            // store the last value of the bool array
            cudaMemcpy(&boolLastVal, dev_bools + n - 1, sizeof(int), cudaMemcpyDeviceToHost);
            checkCUDAErrorFn("copy last bool value to host failed!");

            // scan the bool array
            cudaMemcpy(dev_scan, dev_bools, n * sizeof(int), cudaMemcpyDeviceToDevice);

            // up sweep
            for (int d = 0; d <= log2ceil - 1; ++d) {
#if EFFICIENT   
                // Adjust the grid size based on the depth of the sweep
                dim3 gridSize = (fullSize / (2 << d) + blockSize - 1) / blockSize;
#else
                dim3 gridSize = (fullSize / (2 << d) + blockSize - 1) / blockSize;
#endif

                kernUpSweep << <gridSize, blockSize >> > (fullSize, d, dev_scan);
                checkCUDAErrorFn("up sweep failed!");
            }

            // set the last value to 0
            kernSetLastElementToZero << <1, 1 >> > (fullSize, dev_scan);

            // down sweep
            for (int d = log2ceil - 1; d >= 0; --d) {
#if EFFICIENT   
                // Adjust the grid size based on the depth of the sweep
                dim3 gridSize = (fullSize / (2 << d) + blockSize - 1) / blockSize;
#else
                dim3 gridSize = (fullSize / (2 << d) + blockSize - 1) / blockSize;
#endif

                kernDownSweep << <gridSize, blockSize >> > (fullSize, d, dev_scan);
                checkCUDAErrorFn("down sweep failed");
            }

            // store the last value of the scan results
            cudaMemcpy(&scanLastVal, dev_scan + n - 1, sizeof(int), cudaMemcpyDeviceToHost);
            checkCUDAErrorFn("copy last bool value to host failed!");

            // scatter
            StreamCompaction::Common::kernScatter << <gridSize, blockSize >> > (n, dev_out, dev_in, dev_bools, dev_scan);
            checkCUDAErrorFn("scatter failed!");

            timer().endGpuTimer();

            cudaMemcpy(odata, dev_out, n * sizeof(int), cudaMemcpyDeviceToHost);
            checkCUDAErrorFn("copy dev_out to odata failed!");

            // free memory
            cudaFree(dev_in);
            cudaFree(dev_out);
            cudaFree(dev_bools);
            cudaFree(dev_scan);

            return scanLastVal + boolLastVal;
        }
    }
}

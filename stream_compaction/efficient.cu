#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"
#define BlockSize 256
namespace StreamCompaction {
    namespace Efficient {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }
        __global__ void kernUpSweep(int n, int* odata, int d) {
            int index = threadIdx.x + (blockIdx.x * blockDim.x);
            if (index >= n || (index % (int)powf(2, d + 1) != 0)) {
                return;
            }

            odata[index + (int)powf(2, d + 1) - 1] += odata[index + (int)powf(2, d) - 1];
            
        }

        __global__ void kernDownSweep(int n, int* odata, int d) {
            int index = threadIdx.x + (blockIdx.x * blockDim.x);
            if (index >= n || (index % (int)powf(2, d + 1) != 0)) {
                return;
            }

            int t = odata[index + (int)powf(2, d) - 1];
            odata[index + (int)powf(2, d) - 1] = odata[index + (int)powf(2, d + 1) - 1];
            odata[index + (int)powf(2, d + 1) - 1] += t;
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            int* device_A;


            cudaMalloc((void**)&device_A, n * sizeof(int));
            checkCUDAError("cudaMalloc device_A failed!");

            cudaMemcpy(device_A, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            checkCUDAError("cudaMemcpy cudaMemcpyHostToDevice device_A to idata failed!");

            dim3 blocksPerGrid((n + BlockSize - 1) / BlockSize);

            timer().startGpuTimer();
            // TODO
            for (int d = 0; d <= ilog2ceil(n) - 1; d++) { //Upsweep
                kernUpSweep << <blocksPerGrid, BlockSize >> > (n, device_A, d);

            }

            // Set the root to zero before the downsweep
            int zero = 0;
            cudaMemcpy(&device_A[n - 1], &zero, sizeof(int), cudaMemcpyHostToDevice);

            // Downsweep
            for (int d = ilog2ceil(n) - 1; d >= 0; d--) {
                kernDownSweep << <blocksPerGrid, BlockSize >> > (n, device_A, d);
            }

            timer().endGpuTimer();

            cudaMemcpy(odata, device_A, n * sizeof(int), cudaMemcpyDeviceToHost);
           


            for (int i = 0; i < n; i++) {
                printf("%d ", odata[i]);
            }
            checkCUDAError("cudaMemcpy cudaMemcpyDeviceToHost odata to device_A failed!");

            cudaFree(device_A);
            checkCUDAError("cudaFree device_A failed!");
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

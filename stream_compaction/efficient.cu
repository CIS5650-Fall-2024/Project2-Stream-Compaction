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

        __global__ void kernZero(const int n, int* data) {
            data[n - 1] = 0;
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

        __global__ void kernCompact(int n, int* idata, int* odata) {
            int index = threadIdx.x + (blockIdx.x * blockDim.x);
            if (index >= n) {
                return;
            }
            odata[index] = (idata[index] == 0) ? 0 : 1;
        }

        __global__ void kernScan(int n, int* scan, int* temp) {
            int index = threadIdx.x + blockIdx.x * blockDim.x;
            if (index >= n) {
                return;
            }

            scan[index] = (index > 0) ? scan[index - 1] + temp[index - 1] : 0;
        }

        __global__ void compactKernel(int n, const int* idata, const int* scan, int* odata, const int* temp) {
            int index = blockIdx.x * blockDim.x + threadIdx.x;
            if (index >= n) {
                return;
            }
            if (temp[index] != 0) {
                odata[scan[index]] = idata[index];
            }

        }


        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        int nextPowerOf2(int n) {
            if (n <= 0) return 0;

            int power = 1;
            while (power < n) {
                power *= 2;
            }

            return power;
        }

        void scan(int n, int *odata, const int *idata) {
            int* device_A;

            int paddedSize = nextPowerOf2(n);
            cudaMalloc((void**)&device_A, paddedSize * sizeof(int));
            checkCUDAError("cudaMalloc device_A failed!");


            cudaMemset(device_A + n, 0, (paddedSize - n) * sizeof(int));
            checkCUDAError("device_A cudaMemset failed!");

            

            cudaMemcpy(device_A, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            checkCUDAError("cudaMemcpy cudaMemcpyHostToDevice device_A to idata failed!");

            dim3 blocksPerGrid((paddedSize + BlockSize - 1) / BlockSize);

            timer().startGpuTimer();
            
            for (int d = 0; d <= ilog2ceil(paddedSize) - 1; d++) { //Upsweep
                kernUpSweep << <blocksPerGrid, BlockSize >> > (paddedSize, device_A, d);
            }

            kernZero << <1, 1 >> > (paddedSize, device_A);

            for (int d = ilog2ceil(paddedSize) - 1; d >= 0; d--) { //Downsweep
                kernDownSweep << <blocksPerGrid, BlockSize >> > (paddedSize, device_A, d);
            }


            timer().endGpuTimer();

            
            cudaMemcpy(odata, device_A, n * sizeof(int), cudaMemcpyDeviceToHost);
            


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
            int* device_A, * device_B, * device_Binary, * device_scan;

            int paddedSize = nextPowerOf2(n);
            cudaMalloc((void**)&device_A, paddedSize * sizeof(int));
            checkCUDAError("cudaMalloc device_A failed!");
            cudaMalloc((void**)&device_B, paddedSize * sizeof(int));
            checkCUDAError("cudaMalloc device_B failed!");
            cudaMalloc((void**)&device_Binary, paddedSize * sizeof(int));
            checkCUDAError("cudaMalloc device_Binary failed!");
            cudaMalloc((void**)&device_scan, paddedSize * sizeof(int));
            checkCUDAError("cudaMalloc device_Binary failed!");

            cudaMemcpy(device_A, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            checkCUDAError("cudaMemcpy cudaMemcpyHostToDevice device_A to idata failed!");


            cudaMemset(device_A + n, 0, (paddedSize - n) * sizeof(int));
            checkCUDAError("device_A cudaMemset failed!");
            cudaMemset(device_Binary + n, 0, (paddedSize - n) * sizeof(int));
            checkCUDAError("device_Binary cudaMemset failed!");

            dim3 blocksPerGrid((paddedSize + BlockSize - 1) / BlockSize);

            timer().startGpuTimer();

            for (int d = 0; d <= ilog2ceil(paddedSize) - 1; d++) { //Upsweep
                kernUpSweep << <blocksPerGrid, BlockSize >> > (paddedSize, device_A, d);
            }

            kernZero << <1, 1 >> > (paddedSize, device_A);

            for (int d = ilog2ceil(paddedSize) - 1; d >= 0; d--) { //Downsweep
                kernDownSweep << <blocksPerGrid, BlockSize >> > (paddedSize, device_A, d);
            }
            kernCompact << <blocksPerGrid, BlockSize >> > (paddedSize, device_A, device_Binary);
            kernScan << <blocksPerGrid, BlockSize >> > (paddedSize, device_scan, device_Binary);
            compactKernel << <blocksPerGrid, BlockSize >> > (paddedSize, device_A, device_scan, device_B, device_Binary);
            timer().endGpuTimer();

            cudaMemcpy(odata, device_scan, n * sizeof(int), cudaMemcpyDeviceToHost);

            for (int i = 0; i < n; i++) {
                printf("%d ", odata[i]);
            }


            checkCUDAError("cudaMemcpy cudaMemcpyDeviceToHost odata to device_A failed!");

            cudaFree(device_A);
            checkCUDAError("cudaFree device_A failed!");
            cudaFree(device_B);
            checkCUDAError("cudaFree device_B failed!");
            cudaFree(device_Binary);
            checkCUDAError("cudaFree device_Binary failed!");
            cudaFree(device_scan);
            checkCUDAError("cudaFree device_scan failed!");
            return -1;
        }
    }
}

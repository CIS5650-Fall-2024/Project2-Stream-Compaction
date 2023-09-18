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
        __global__ void kernUpCopy(int n, int* idata, int* odata) {
            int index = threadIdx.x + (blockIdx.x * blockDim.x);
            if (index >= n) {
                return;
            }

            if (odata[index] != idata[index]) {
                odata[index] = idata[index];
            }

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
            int* device_idata, * device_odata, * device_bool, * device_scan;

            int paddedSize = nextPowerOf2(n);
            cudaMalloc((void**)&device_idata, paddedSize * sizeof(int));
            checkCUDAError("cudaMalloc device_idata failed!");
            cudaMalloc((void**)&device_odata, paddedSize * sizeof(int));
            checkCUDAError("cudaMalloc device_odata failed!");
            cudaMalloc((void**)&device_bool, paddedSize * sizeof(int));
            checkCUDAError("cudaMalloc device_bool failed!");
            cudaMalloc((void**)&device_scan, paddedSize * sizeof(int));
            checkCUDAError("cudaMalloc device_scan failed!");

            cudaMemcpy(device_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            checkCUDAError("cudaMemcpy cudaMemcpyHostToDevice device_idata to idata failed!");


            cudaMemset(device_idata + n, 0, (paddedSize - n) * sizeof(int));
            checkCUDAError("device_idata cudaMemset failed!");
            cudaMemset(device_scan + n, 0, (paddedSize - n) * sizeof(int));
            checkCUDAError("device_scan cudaMemset failed!");

            dim3 blocksPerGrid((paddedSize + BlockSize - 1) / BlockSize);

            timer().startGpuTimer();
            StreamCompaction::Common::kernMapToBoolean << <blocksPerGrid, BlockSize >> > (n, device_bool, device_idata);
            kernUpCopy << <blocksPerGrid, BlockSize >> > (n, device_bool, device_scan);


            for (int d = 0; d <= ilog2ceil(paddedSize) - 1; d++) { //Upsweep
                kernUpSweep << <blocksPerGrid, BlockSize >> > (paddedSize, device_scan, d);
            }

            kernZero << <1, 1 >> > (paddedSize, device_scan);

            for (int d = ilog2ceil(paddedSize) - 1; d >= 0; d--) { //Downsweep
                kernDownSweep << <blocksPerGrid, BlockSize >> > (paddedSize, device_scan, d);
            }

            StreamCompaction::Common::kernScatter << <blocksPerGrid, BlockSize >> > (paddedSize, device_odata, device_idata, device_bool, device_scan);
            timer().endGpuTimer();

            cudaMemcpy(odata, device_odata, n * sizeof(int), cudaMemcpyDeviceToHost);
            int finalSize;
            cudaMemcpy(&finalSize, device_scan + paddedSize - 1, sizeof(int), cudaMemcpyDeviceToHost);
            checkCUDAError("cudaMemcpy cudaMemcpyDeviceToHost odata to device_idata failed!");

            cudaFree(device_idata);
            checkCUDAError("cudaFree device_idata failed!");
            cudaFree(device_odata);
            checkCUDAError("cudaFree device_odata failed!");
            cudaFree(device_bool);
            checkCUDAError("cudaFree device_bool failed!");
            cudaFree(device_scan);
            checkCUDAError("cudaFree device_scan failed!");
            return finalSize;
        }
    }
}

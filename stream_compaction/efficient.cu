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

        __global__ void kernUpSweep(int n, int d, int* data) {
            int index = threadIdx.x + (blockIdx.x * blockDim.x);         
            if (index >= n) return;

            int a = 1 << d;
            int b = 1 << (d + 1);
            if (index % b == 0)
            {
                data[index + b - 1] += data[index + a - 1];
            }  
        }

        __global__ void kernDownSweep(int n, int d, int* data){
            int index = threadIdx.x + (blockIdx.x * blockDim.x);         
            if (index >= n) return;

            int a = 1 << d;
            int b = 1 << (d + 1);
            if (index % b == 0)
            {
                int temp = data[index + b - 1];
                data[index + b - 1] += data[index + a - 1];
                data[index + a - 1] = temp;
            }  
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            
            // TODO
            dim3 fullBlocksPerGrid((n + blockSize - 1) / blockSize);
            int size = pow(2, ilog2ceil(n));

            int* scan_array;
            cudaMalloc((void**)&scan_array, size * sizeof(int));
            cudaMemset(scan_array, 0, size * sizeof(int));
            cudaMemcpy(scan_array, idata, size * sizeof(int), cudaMemcpyHostToDevice);

            timer().startGpuTimer();
            // Up sweep
            for (int i = 0; i < ilog2ceil(n); i++)
            {                       
                kernUpSweep <<<fullBlocksPerGrid, blockSize>>> (size, i, scan_array);
                checkCUDAError("kernUpSweep fails.");
            }
            cudaMemset(scan_array + size - 1, 0, sizeof(int));

            // Down sweep
            for (int i = ilog2ceil(n) - 1; i >= 0; i--)
            {                       
                kernDownSweep <<<fullBlocksPerGrid, blockSize>>> (size, i, scan_array);
                checkCUDAError("kernDownSweep fails.");
            }
            timer().endGpuTimer();

            cudaMemcpy(odata, scan_array, n * sizeof(int), cudaMemcpyDeviceToHost);
            cudaFree(scan_array);
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
            
            // TODO
            dim3 fullBlocksPerGrid((n + blockSize - 1) / blockSize);
            int size = pow(2, ilog2ceil(n));

            int* iarray;
            int* bool_array;
            int* scan_array;
            cudaMalloc((void**)&iarray, size * sizeof(int));
            cudaMalloc((void**)&bool_array, size * sizeof(int));
            cudaMalloc((void**)&scan_array, size * sizeof(int));
            cudaMemcpy(iarray, idata, size * sizeof(int), cudaMemcpyHostToDevice);

            timer().startGpuTimer();

            // map to bool
            Common::kernMapToBoolean<<<fullBlocksPerGrid, blockSize>>>(size, bool_array, iarray);
            cudaMemcpy(scan_array, bool_array, size * sizeof(int), cudaMemcpyDeviceToDevice);

            // up sweep
            for (int i = 0; i < ilog2ceil(n); i++) {
                kernUpSweep <<<fullBlocksPerGrid, blockSize>>> (size, i, scan_array);
                checkCUDAError("kernUpSweep fails.");
            }
            cudaMemset(scan_array + size - 1, 0, sizeof(int));

            // down sweep
            for (int i = ilog2ceil(n) - 1; i >= 0; i--) {
                kernDownSweep <<<fullBlocksPerGrid, blockSize>>> (size, i, scan_array);
                checkCUDAError("kernDownSweep fails.");
            }

            // scatter
            int* oarray;
            cudaMalloc((void**)&oarray, size * sizeof(int));
            Common::kernScatter <<<fullBlocksPerGrid, blockSize>>>(size, oarray, iarray, bool_array, scan_array);

            timer().endGpuTimer();

            int a = 0;
            int b = 0;
            cudaMemcpy(&a, scan_array + size - 1, sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(&b, bool_array + size - 1, sizeof(int), cudaMemcpyDeviceToHost);
            int cnt = a + b;
            cudaMemcpy(odata, oarray, cnt * sizeof(int), cudaMemcpyDeviceToHost);

            cudaFree(iarray);
            cudaFree(oarray);
            cudaFree(bool_array);
            cudaFree(scan_array);

            return cnt;
        }
    }
}

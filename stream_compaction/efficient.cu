#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

#define blockSize 256

namespace StreamCompaction {
    namespace Efficient {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        __global__ void kernUpSweep(int n, int d, int* data) 
        {
            int k = (blockIdx.x * blockDim.x) + threadIdx.x;

            int powerd = 1 << d;
            int powerdp1 = 1 << (d + 1);

            if (k >= n || k % powerdp1) return;

            data[k + powerdp1 - 1] = data[k + powerd - 1] + data[k + powerdp1 - 1];
        }

        __global__ void kernDownSweep(int n, int d, int* data) 
        {
            int k = (blockIdx.x * blockDim.x) + threadIdx.x;           

            int powerd = 1 << d;
            int powerdp1 = 1 << (d + 1);

            if (k >= n || k % powerdp1 || k + powerdp1 - 1 >= n) return;

            int t = data[k + powerd - 1];
            data[k + powerd - 1] = data[k + powerdp1 - 1];
            data[k + powerdp1 - 1] = t + data[k + powerdp1 - 1];
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) 
        {
            int N = 1 << ilog2ceil(n);

            int* dev_data;

            cudaMalloc((void**)&dev_data, N * sizeof(int));
            checkCUDAErrorFn("cudaMalloc dev_data failed!");
            cudaMemset(dev_data, 0, N * sizeof(int));
            checkCUDAErrorFn("cudaMemset dev_data to 0 failed!");
            cudaMemcpy(dev_data, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            checkCUDAErrorFn("cudaMemcpy idata to dev_data failed!");

            dim3 fullBlocksPerGrid((N + blockSize - 1) / blockSize);

            timer().startGpuTimer();
            // TODO
            for (int d = 0; d < ilog2ceil(N); d++)
            {
                kernUpSweep<<<fullBlocksPerGrid, blockSize>>>(N, d, dev_data);
            }

            cudaMemset(dev_data + N - 1, 0, sizeof(int));
            for (int d = ilog2ceil(N) - 1; d >= 0; d--) 
            {
                kernDownSweep<<<fullBlocksPerGrid, blockSize>>>(N, d, dev_data);
            }
            timer().endGpuTimer();

            cudaMemcpy(odata, dev_data, n * sizeof(int), cudaMemcpyDeviceToHost);
            checkCUDAErrorFn("cudaMemcpy dev_data to odata failed!");

            cudaFree(dev_data);
            checkCUDAErrorFn("cudaFree dev_data failed!");
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
        int compact(int n, int *odata, const int *idata) 
        {
            int N = 1 << ilog2ceil(n);

            int* dev_bools;
            int* dev_data;
            int* dev_idata;
            int* dev_odata;

            cudaMalloc((void**)&dev_bools, n * sizeof(int));
            checkCUDAErrorFn("cudaMalloc dev_bools failed!");
            cudaMalloc((void**)&dev_data, N * sizeof(int));
            checkCUDAErrorFn("cudaMalloc dev_data failed!");
            cudaMalloc((void**)&dev_idata, n * sizeof(int));
            checkCUDAErrorFn("cudaMalloc dev_idata failed!");
            cudaMalloc((void**)&dev_odata, n * sizeof(int));
            checkCUDAErrorFn("cudaMalloc dev_odata failed!");

            cudaMemset(dev_data, 0, N * sizeof(int));
            checkCUDAErrorFn("cudaMemset dev_data failed!");
            cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            checkCUDAErrorFn("cudaMempcy idata to dev_idata failed!");

            dim3 fullBlocksPerGrid((n + blockSize - 1) / blockSize);

            timer().startGpuTimer();
            // TODO
            StreamCompaction::Common::kernMapToBoolean<<<fullBlocksPerGrid, blockSize>>>(n, dev_bools, dev_idata);

            cudaMemcpy(dev_data, dev_bools, n * sizeof(int), cudaMemcpyDeviceToDevice);
            checkCUDAErrorFn("cudaMempcy dev_bools to dev_data failed!");

            for (int d = 0; d < ilog2ceil(N); d++)
            {
                kernUpSweep<<<fullBlocksPerGrid, blockSize>>>(N, d, dev_data);
            }

            cudaMemset(dev_data + N - 1, 0, sizeof(int));
            for (int d = ilog2ceil(N) - 1; d >= 0; d--)
            {
                kernDownSweep<<<fullBlocksPerGrid, blockSize>>>(N, d, dev_data);
            }

            StreamCompaction::Common::kernScatter<<<fullBlocksPerGrid, blockSize>>>(n, dev_odata, dev_idata, dev_bools, dev_data);

            timer().endGpuTimer();

            int count = 0;
            cudaMemcpy(&count, dev_data + N - 1, sizeof(int), cudaMemcpyDeviceToHost);
            checkCUDAErrorFn("cudaMempcy count failed!");
            cudaMemcpy(odata, dev_odata, n * sizeof(int), cudaMemcpyDeviceToHost);
            checkCUDAErrorFn("cudaMempcy dev_odata to odata failed!");

            cudaFree(dev_bools);
            checkCUDAErrorFn("cudaFree dev_bools failed!");
            cudaFree(dev_data);
            checkCUDAErrorFn("cudaFree dev_data failed!");
            cudaFree(dev_idata);
            checkCUDAErrorFn("cudaFree dev_idata failed!");
            cudaFree(dev_odata);
            checkCUDAErrorFn("cudaFree dev_odata failed!");

            return count;
        }
    }
}

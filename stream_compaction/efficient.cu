#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

#define blockSize 256
#define OPTIMIZED 1
#define TIMESCAN 1

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

            data[k + powerdp1 - 1] += data[k + powerd - 1];
        }

        __global__ void kernDownSweep(int n, int d, int* data) 
        {
            int k = (blockIdx.x * blockDim.x) + threadIdx.x;           

            int powerd = 1 << d;
            int powerdp1 = 1 << (d + 1);

            if (k >= n || k % powerdp1 || k + powerdp1 - 1 >= n) return;

            int t = data[k + powerd - 1];
            data[k + powerd - 1] = data[k + powerdp1 - 1];
            data[k + powerdp1 - 1] += t;
        }

        __global__ void kernOptUpSweep(int n, int d, int offset, int* data)
        {
            int k = (blockIdx.x * blockDim.x) + threadIdx.x;

            if (k >= n || k >= d) return;

            int i = offset * (2 * k + 1) - 1;
            int j = offset * (2 * k + 2) - 1;

            data[j] += data[i];
        }

        __global__ void kernOptDownSweep(int n, int d, int offset, int* data)
        {
            int k = (blockIdx.x * blockDim.x) + threadIdx.x;

            if (k >= n || k >= d) return;

            int i = offset * (2 * k + 1) - 1;
            int j = offset * (2 * k + 2) - 1;

            int t = data[i];
            data[i] = data[j];
            data[j] += t;
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
#if OPTIMIZED
            int offset = 1;
#endif

#if TIMESCAN
            timer().startGpuTimer();
#endif
            // TODO
#if OPTIMIZED
            for (int d = N >> 1; d > 0; d >>= 1) 
            {
                fullBlocksPerGrid = dim3((d + blockSize - 1) / blockSize);
                kernOptUpSweep<<<fullBlocksPerGrid, blockSize>>>(N, d, offset, dev_data);
                offset <<= 1;
            }

#else
            for (int d = 0; d < ilog2ceil(N); d++)
            {
                kernUpSweep<<<fullBlocksPerGrid, blockSize>>>(N, d, dev_data);
            }
#endif

            cudaMemset(dev_data + N - 1, 0, sizeof(int));
#if OPTIMIZED
            for (int d = 1; d < N; d <<= 1)
            {
                offset >>= 1;
                fullBlocksPerGrid = dim3((d + blockSize - 1) / blockSize);
                kernOptDownSweep<<<fullBlocksPerGrid, blockSize>>>(N, d, offset, dev_data);
            }
#else
            for (int d = ilog2ceil(N) - 1; d >= 0; d--) 
            {
                kernDownSweep<<<fullBlocksPerGrid, blockSize>>>(N, d, dev_data);
            }
#endif

#if TIMESCAN
            timer().endGpuTimer();
#endif

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

        __global__ void kernComputeEArray(int n, int bit, int* edata, const int* idata)
        {
            int index = (blockIdx.x * blockDim.x) + threadIdx.x;
            if (index >= n) return;

            edata[index] = !((idata[index] >> bit) & 1);
        }

        __global__ void kernComputeTArray(int n, int totalFalses, int* tdata, const int* fdata)
        {
            int index = (blockIdx.x * blockDim.x) + threadIdx.x;
            if (index >= n) return;

            tdata[index] = index - fdata[index] + totalFalses;
        }

        __global__ void kernComputeDArray(int n, int* ddata, const int* edata, const int* tdata, const int* fdata)
        {
            int index = (blockIdx.x * blockDim.x) + threadIdx.x;
            if (index >= n) return;

            ddata[index] = edata[index] ? fdata[index] : tdata[index];
        }

        __global__ void kernScatter(int n, int* ddata, int* odata, int* idata) 
        {
            int index = (blockIdx.x * blockDim.x) + threadIdx.x;
            if (index >= n) return;

            odata[ddata[index]] = idata[index];
        }

        void radixSort(int n, int* odata, const int* idata)
        {
            int* dev_edata;
            int* dev_fdata;
            int* dev_tdata;
            int* dev_ddata;

            int* dev_idata;
            int* dev_odata;

            cudaMalloc((void**)&dev_edata, n * sizeof(int));
            checkCUDAErrorFn("cudaMalloc dev_edata failed!");
            cudaMalloc((void**)&dev_fdata, n * sizeof(int));
            checkCUDAErrorFn("cudaMalloc dev_fdata failed!");
            cudaMalloc((void**)&dev_tdata, n * sizeof(int));
            checkCUDAErrorFn("cudaMalloc dev_tdata failed!");
            cudaMalloc((void**)&dev_ddata, n * sizeof(int));
            checkCUDAErrorFn("cudaMalloc dev_ddata failed!");

            cudaMalloc((void**)&dev_idata, n * sizeof(int));
            checkCUDAErrorFn("cudaMalloc dev_idata failed!");
            cudaMalloc((void**)&dev_odata, n * sizeof(int));
            checkCUDAErrorFn("cudaMalloc dev_odata failed!");

            cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            checkCUDAErrorFn("cudaMempcy idata to dev_idata failed!");

            dim3 fullBlocksPerGrid((n + blockSize - 1) / blockSize);

            int bnum = ilog2ceil(*(std::max_element(idata, idata + n)));
            
            timer().startGpuTimer();
            for (int d = 0; d < bnum; d++) 
            {
                // Step1: Compute e array
                kernComputeEArray<<<fullBlocksPerGrid, blockSize>>>(n, d, dev_edata, dev_idata);
                // Step2: Scan e
                scan(n, dev_fdata, dev_edata);
                // Step3: Compute totalFalse
                int e_last;
                int f_last;
                cudaMemcpy(&e_last, dev_edata + n - 1, sizeof(int), cudaMemcpyDeviceToHost);
                cudaMemcpy(&f_last, dev_fdata + n - 1, sizeof(int), cudaMemcpyDeviceToHost);
                int totalFalses = e_last + f_last;
                // Step4: Compute t
                kernComputeTArray<<<fullBlocksPerGrid, blockSize>>>(n, totalFalses, dev_tdata, dev_fdata);
                // Step5: scatter
                kernComputeDArray<<<fullBlocksPerGrid, blockSize>>>(n, dev_ddata, dev_edata, dev_tdata, dev_fdata);
                kernScatter<<<fullBlocksPerGrid, blockSize>>>(n, dev_ddata, dev_odata, dev_idata);
                cudaMemcpy(dev_idata, dev_odata, n * sizeof(int), cudaMemcpyDeviceToDevice);
                checkCUDAErrorFn("cudaMempcy dev_odata to dev_idata failed!");
            }
            timer().endGpuTimer();

            cudaMemcpy(odata, dev_odata, n * sizeof(int), cudaMemcpyDeviceToHost);
            checkCUDAErrorFn("cudaMempcy dev_odata to odata failed!");

            cudaFree(dev_edata);
            checkCUDAErrorFn("cudaFree dev_edata failed!");
            cudaFree(dev_fdata);
            checkCUDAErrorFn("cudaFree dev_fdata failed!");
            cudaFree(dev_tdata);
            checkCUDAErrorFn("cudaFree dev_tdata failed!");
            cudaFree(dev_ddata);
            checkCUDAErrorFn("cudaFree dev_ddata failed!");
            cudaFree(dev_idata);
            checkCUDAErrorFn("cudaFree dev_idata failed!");
            cudaFree(dev_odata);
            checkCUDAErrorFn("cudaFree dev_odata failed!");
        }
    }
}

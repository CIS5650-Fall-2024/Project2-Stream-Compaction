#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"
#define BLOCK_SIZE 1024
namespace StreamCompaction {
    namespace Efficient {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        __global__ void kernUpSweep(int N, int* array, int two_d_plus_1)
        {
            int index = blockIdx.x * blockDim.x + threadIdx.x;
            if (index >= N) return;
            int k = index * two_d_plus_1;
            array[k + two_d_plus_1 - 1] += array[k + (two_d_plus_1 >> 1) - 1];
        }

        __global__ void kernDownSweep(int N, int* array, int two_d_plus_1)
        {
            int index = blockIdx.x * blockDim.x + threadIdx.x;
            if (index >= N) return;
            int k = index * two_d_plus_1;
            if (N == 1) array[k + two_d_plus_1 - 1] = 0;
            int two_d = two_d_plus_1 >> 1;
            int tmp = array[k + two_d - 1];
            array[k + two_d - 1] = array[k + two_d_plus_1 - 1];
            array[k + two_d_plus_1 - 1] += tmp;
        }

        __global__ void kernCompact(int N, int* prefix, int* input, int* output)
        {
            int index = blockIdx.x * blockDim.x + threadIdx.x;
            if (index >= N) return;
            if (input[index])
            {
                output[prefix[index]] = input[index];
            }
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int* odata, const int* idata) {
            // TODO
            int log2n = ilog2ceil(n);
            size_t N = (size_t)1 << log2n;
            int* dev1;
            cudaMalloc((void**)&dev1, N * sizeof(int));
            cudaMemset(dev1, 0, N * sizeof(int));
            cudaMemcpy(dev1, idata, sizeof(int) * n, cudaMemcpyHostToDevice);
            cudaDeviceSynchronize();
            nvtxRangePushA("Work efficient scan");
            timer().startGpuTimer();
            
            for (int d = 0; d <= log2n - 1; d++)
            {
                size_t two_d_plus_1 = ((size_t)1 << (d + 1));
                int numThreads = N / two_d_plus_1;
                int numBlocks = (numThreads + BLOCK_SIZE - 1) / BLOCK_SIZE;
                if (numThreads >= BLOCK_SIZE)
                    kernUpSweep << <numBlocks, BLOCK_SIZE >> > (numThreads, dev1, two_d_plus_1);
                else
                    kernUpSweep << <1, numThreads >> > (numThreads, dev1, two_d_plus_1);
            }
            for (int d = log2n - 1; d >= 0; d--)
            {
                size_t two_d_plus_1 = ((size_t)1 << (d + 1));
                int numThreads = N / two_d_plus_1;
                int numBlocks = (numThreads + BLOCK_SIZE - 1) / BLOCK_SIZE;
                if (numThreads >= BLOCK_SIZE)
                    kernDownSweep << <numBlocks, BLOCK_SIZE >> > (numThreads, dev1, two_d_plus_1);
                else
                    kernDownSweep << <1, numThreads >> > (numThreads, dev1, two_d_plus_1);
            }
            timer().endGpuTimer();
            nvtxRangePop();
            cudaMemcpy(odata, dev1, sizeof(int) * n, cudaMemcpyDeviceToHost);
            cudaFree(dev1);

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
        int compact(int n, int* odata, const int* idata) {

            // TODO
            int log2n = ilog2ceil(n);
            size_t N = (size_t)1 << log2n;
            int* dev1, * dev2, * dev3;;
            cudaMalloc((void**)&dev1, n * sizeof(int));
            cudaMalloc((void**)&dev2, N * sizeof(int));
            cudaMemset(dev2, 0, N * sizeof(int));
            cudaMemcpy(dev1, idata, sizeof(int) * n, cudaMemcpyHostToDevice);
            cudaMalloc((void**)&dev3, n * sizeof(int));
            nvtxRangePushA("Work efficient compact");
            timer().startGpuTimer();
            Common::kernMapToBoolean << <(n + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE >> > (n, dev2, dev1);
            for (int d = 0; d <= log2n - 1; d++)
            {
                size_t two_d_plus_1 = ((size_t)1 << (d + 1));
                int numThreads = N / two_d_plus_1;
                int numBlocks = (numThreads + BLOCK_SIZE - 1) / BLOCK_SIZE;
                kernUpSweep << <numBlocks, BLOCK_SIZE >> > (numThreads, dev2, two_d_plus_1);
            }
            for (int d = log2n - 1; d >= 0; d--)
            {
                size_t two_d_plus_1 = ((size_t)1 << (d + 1));
                int numThreads = N / two_d_plus_1;
                int numBlocks = (numThreads + BLOCK_SIZE - 1) / BLOCK_SIZE;
                kernDownSweep << <numBlocks, BLOCK_SIZE >> > (numThreads, dev2, two_d_plus_1);
            }
            kernCompact << <(n + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE >> > (n, dev2, dev1, dev3);
            timer().endGpuTimer();
            nvtxRangePop();
            int excnt;
            cudaMemcpy(&excnt, dev2 + n - 1, sizeof(int), cudaMemcpyDeviceToHost);
            int cnt = excnt + !!(idata[n - 1]);
            cudaMemcpy(odata, dev3, sizeof(int) * n, cudaMemcpyDeviceToHost);
            cudaFree(dev1);
            cudaFree(dev2);
            cudaFree(dev3);
            return cnt;
        }
    }
}

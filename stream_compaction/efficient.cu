#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

#define NUM_BANKS 32 
#define LOG_NUM_BANKS 5 
#define CONFLICT_FREE_OFFSET(n) ((n) >> NUM_BANKS + (n) >> (2 * LOG_NUM_BANKS)) 

namespace StreamCompaction {
    namespace Efficient {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        __global__ void kernUpSweep(int N, int offset, int* data) {
            size_t k = (threadIdx.x + (blockIdx.x * blockDim.x)) * 2 * offset;
            if (k >= N) {
                return;
            }
            data[k + 2 * offset - 1] += data[k + offset - 1];
        }

        __global__ void kernDownSweep(int N, int offset, int* data) {
            size_t k = (threadIdx.x + (blockIdx.x * blockDim.x)) * 2 * offset;
            if (k >= N) {
                return;
            }
            size_t left = k + offset - 1;
            size_t right = k + 2 * offset - 1;
            int temp = data[left];
            data[left] = data[right];
            data[right] += temp;
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            int N = 1 << ilog2ceil(n);
            int* dev_data;
            cudaMalloc((void**)&dev_data, N * sizeof(int));
            checkCUDAErrorFn("malloc dev_data failed");
            cudaMemcpy(dev_data, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            checkCUDAErrorFn("copy idata to dev_data failed");
            dim3 B((N + BLOCK_SIZE - 1) / BLOCK_SIZE);
            timer().startGpuTimer();
            // TODO
            for (int i = 1; i < N; i <<= 1) {
                kernUpSweep << <B, BLOCK_SIZE >> > (N, i, dev_data);
                B.x = std::max(B.x >> 1, 1U);
            }
            cudaMemset(dev_data + N - 1, 0, sizeof(int));
            checkCUDAErrorFn("set dev_data root to 0 failed");
            for (int i = N >> 1; i > 0; i >>= 1) {
                B.x = (N / (2 * i) + BLOCK_SIZE - 1) / BLOCK_SIZE;
                kernDownSweep << <B, BLOCK_SIZE >> > (N, i, dev_data);
            }
            timer().endGpuTimer();
            cudaMemcpy(odata, dev_data, n * sizeof(int), cudaMemcpyDeviceToHost);
            checkCUDAErrorFn("copy dev_data to odata failed");
            cudaFree(dev_data);
            checkCUDAErrorFn("cudaFree dev_data failed");
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
            int N = 1 << ilog2ceil(n);
            int* dev_idata;
            int* dev_bools;
            int* dev_index;
            int* dev_odata;
            int compressedLength;

            cudaMalloc((void**)&dev_idata, N * sizeof(int));
            checkCUDAErrorFn("malloc dev_idata failed");
            cudaMemset(dev_idata, 0, N * sizeof(int));
            checkCUDAErrorFn("zeroing dev_idata failed");
            cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            checkCUDAErrorFn("copy idata to dev_idata failed");
            cudaMalloc((void**)&dev_bools, N * sizeof(int));
            checkCUDAErrorFn("malloc dev_bools failed");
            cudaMalloc((void**)&dev_index, N * sizeof(int));
            checkCUDAErrorFn("malloc dev_index failed");

            cudaMalloc((void**)&dev_odata, N * sizeof(int));
            checkCUDAErrorFn("malloc dev_odata failed");
            
            dim3 fullBlocksPerGrid((N + BLOCK_SIZE - 1) / BLOCK_SIZE);
            dim3 B((N + BLOCK_SIZE - 1) / BLOCK_SIZE);
            timer().startGpuTimer();
            // TODO
            StreamCompaction::Common::kernMapToBoolean<<<fullBlocksPerGrid, BLOCK_SIZE>>>(N, dev_bools, dev_idata);
            cudaMemcpy(dev_index, dev_bools, N * sizeof(int), cudaMemcpyDeviceToDevice);
            for (int i = 1; i < N; i <<= 1) {
                kernUpSweep << <B, BLOCK_SIZE >> > (N, i, dev_index);
                B.x = std::max(B.x >> 1, 1U);
            }
            cudaMemset(dev_index + N - 1, 0, sizeof(int));
            checkCUDAErrorFn("set dev_bools root to 0 failed");
            for (int i = N >> 1; i > 0; i >>= 1) {
                B.x = (N / i + BLOCK_SIZE - 1) / BLOCK_SIZE;
                kernDownSweep << <B, BLOCK_SIZE >> > (N, i, dev_index);
            }
            cudaMemcpy(&compressedLength, dev_index + N - 1, sizeof(int), cudaMemcpyDeviceToHost);
            checkCUDAErrorFn("copy dev_index[N-1] to compressedLength failed");
            StreamCompaction::Common::kernScatter << <fullBlocksPerGrid, BLOCK_SIZE >> > (n, dev_odata, dev_idata, dev_bools, dev_index);
            timer().endGpuTimer();
            cudaMemcpy(odata, dev_odata, compressedLength * sizeof(int), cudaMemcpyDeviceToHost);
            checkCUDAErrorFn("copy dev_odata to odata failed");

            cudaFree(dev_idata);
            checkCUDAErrorFn("cudaFree dev_idata failed");
            cudaFree(dev_odata);
            checkCUDAErrorFn("cudaFree dev_odata failed");
            cudaFree(dev_bools);
            checkCUDAErrorFn("cudaFree dev_bools failed");
            cudaFree(dev_index);
            checkCUDAErrorFn("cudaFree dev_index failed");

            return compressedLength;
        }
    }
}

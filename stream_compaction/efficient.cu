#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

namespace StreamCompaction {
    namespace Efficient {
        bool disableScanTimer = false;
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        //__device__ int ilog2ceil(int x) {
        //    return 32 - __clz(x - 1);
        //}

        __global__ void kernNaiveWorkEfficientScanUpSweep(int n, int d, int* data) {
            int k = (blockIdx.x * blockDim.x) + threadIdx.x;
            if (k >= n) return;

            int pow2_d = 1 << d;
            int pow2_dp1 = pow2_d << 1;

            if ((k & (pow2_dp1 - 1)) == 0) {
                data[k + pow2_dp1 - 1] += data[k + pow2_d - 1];
            }
        }

        __global__ void kernNaiveWorkEfficientScanDownSweep(int n, int d, int* data) {
            int k = (blockIdx.x * blockDim.x) + threadIdx.x;
            if (k >= n) return;

            int pow2_d = 1 << d;
            int pow2_dp1 = pow2_d << 1;

            if ((k & (pow2_dp1 - 1)) == 0) {
                int t = data[k + pow2_d - 1];
                data[k + pow2_d - 1] = data[k + pow2_dp1 - 1];
                data[k + pow2_dp1 - 1] += t;
            }
        }

        //__global__ void kernNaiveWorkEfficientScanUpSweep(int n, int d, int* data) {
        //    extern __shared__ int partialSum[];

        //    int index = (blockIdx.x * blockDim.x) + threadIdx.x;
        //    //if (index >= n) return;

        //    partialSum[index >> 1] = data[index];

        //    int offset = 1;

        //    for (int d = n >> 1; d > 0; d >>= 1) {
        //        __syncthreads();
        //        if (index < d) {
        //            int ai = offset * (2 * index + 1) - 1;
        //            int bi = offset * (2 * index + 2) - 1;
        //            partialSum[bi - 1] += partialSum[ai + blockDim.x - 1];
        //        }
        //        offset >>= 1;

        //    }

        //}
#if 0
        __global__ void kernCustomWorkEfficientScan(int n, int* data) {
            extern __shared__ int partialSum[];

            int index = (blockIdx.x * blockDim.x) + threadIdx.x;

            // for debugging
            int* p = partialSum;


            partialSum[threadIdx.x] = data[index];
            __syncthreads();
           
            // upsweep
            for (int d = 1; d <= ilog2ceil(blockDim.x); d++) {
                int active = __ffs(threadIdx.x + 1);
                if (__ffs(threadIdx.x + 1) > d) {
                    int otherIdx = threadIdx.x - (1 << (d - 1));
                    partialSum[threadIdx.x] += partialSum[otherIdx];
                }
                __syncthreads();
            }

            // downsweep
            if (threadIdx.x == 0) partialSum[blockDim.x - 1] = 0;
            __syncthreads();
            for (int d = ilog2ceil(blockDim.x); d >= 1; d--) {
                if (__ffs(threadIdx.x + 1) > d) {
                    int otherIdx = threadIdx.x - (1 << (d - 1));
                    int tmp = partialSum[otherIdx];
                    partialSum[otherIdx] = partialSum[threadIdx.x];
                    partialSum[threadIdx.x] += tmp;
                }
                __syncthreads();
            }

            data[index] = partialSum[threadIdx.x];

        }
#endif
        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            int nextPow2_n = 1 << ilog2ceil(n);
            int* dev_data;
            cudaMalloc((void**)&dev_data, nextPow2_n * sizeof(int));
            cudaMemset(dev_data, 0, nextPow2_n * sizeof(int));
            cudaMemcpy(dev_data, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            checkCUDAError("cudaMalloc, cudaMemset, cudaMemcpy dev_data failed!");
            cudaDeviceSynchronize();
            if (!disableScanTimer) timer().startGpuTimer();
            // ----------------------------------
            // TODO
            int blockSize = 32; // gauranteed no bank conflicts
            dim3 threadsPerBlock(blockSize);
            dim3 fullBlocksPerGrid((nextPow2_n + blockSize - 1) / blockSize);

            // upsweep
            for (int d = 0; d < ilog2ceil(n); d++) {
                kernNaiveWorkEfficientScanUpSweep<<<fullBlocksPerGrid, threadsPerBlock>>>(nextPow2_n, d, dev_data);
            }
            cudaMemset(dev_data + nextPow2_n - 1, 0, sizeof(int));
            // downsweep
            for (int d = ilog2ceil(n) - 1; d >= 0; d--) {
                kernNaiveWorkEfficientScanDownSweep<<<fullBlocksPerGrid, threadsPerBlock>>>(nextPow2_n, d, dev_data);
            }
            // ----------------------------------
            if (!disableScanTimer) timer().endGpuTimer();
            // dev_data now contains an exclusive scan
            cudaMemcpy(odata, dev_data, n * sizeof(int), cudaMemcpyDeviceToHost);
            checkCUDAError("cudaMemcpy dev_data failed!");
            cudaFree(dev_data);
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
            int* dev_idata;
            cudaMalloc((void**)&dev_idata, n * sizeof(int));
            cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            checkCUDAError("cudaMalloc, cudaMemcpy dev_idata failed!");
            int* dev_odata;
            cudaMalloc((void**)&dev_odata, n * sizeof(int));
            checkCUDAError("cudaMalloc dev_odata failed!");
            int* dev_bools;
            cudaMalloc((void**)&dev_bools, n * sizeof(int));
            checkCUDAError("cudaMalloc dev_bools failed!");
            int* dev_indices;
            cudaMalloc((void**)&dev_indices, n * sizeof(int));
            checkCUDAError("cudaMalloc dev_indices failed!");
            cudaDeviceSynchronize();
            disableScanTimer = true;
            timer().startGpuTimer();
            // ----------------------------------
            // TODO
            int blockSize = 32; // gauranteed no bank conflicts
            dim3 threadsPerBlock(blockSize);
            dim3 fullBlocksPerGrid((n + blockSize - 1) / blockSize);
            StreamCompaction::Common::kernMapToBoolean<<<fullBlocksPerGrid, threadsPerBlock>>>(n, dev_bools, dev_idata);
            scan(n, dev_indices, dev_bools);
            StreamCompaction::Common::kernScatter<<<fullBlocksPerGrid, threadsPerBlock>>>(n, dev_odata, dev_idata, dev_bools, dev_indices);
            // ----------------------------------
            timer().endGpuTimer();
            disableScanTimer = false;
            int count;
            cudaMemcpy(&count, dev_indices + n - 1, sizeof(int), cudaMemcpyDeviceToHost);
            count += idata[n - 1] == 0 ? 0 : 1;
            cudaMemcpy(odata, dev_odata, n * sizeof(int), cudaMemcpyDeviceToHost);
            checkCUDAError("cudaMemcpy failed!");
            cudaFree(dev_idata);
            cudaFree(dev_odata);
            cudaFree(dev_bools);
            cudaFree(dev_indices);
            return count;
        }
    }
}

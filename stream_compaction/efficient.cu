#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

/** default naive implementation from lecture (disable all defs)
 *  also three improved implementations of work-efficient scan
 *      1. GEM3 base
 *      2. GEM3 full impl
 *      3. GEM3 full impl with recursive scan (experimental)
 */
#define GEM3 // tested, 5 ~ 10 times slower than thrust
#define FULL // tested, 1.5 ~ 3 times slower than thrust
//#define RECUR // works but not fully tested, this should work well for extremely large array, like size >= 2^32

namespace StreamCompaction {
    namespace Efficient {
        bool disableScanTimer = false;
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

#ifdef GEM3 // ref: https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda

#define BLOCK_SIZE 128
#define THREADS_PER_BLOCK BLOCK_SIZE
#define NUM_BLOCKS(n) (((n) + BLOCK_SIZE - 1) / BLOCK_SIZE)

        __global__ void kernGEM3WarpReorgWorkEfficientScanUpSweep(int n, int d, int offset, int* data) {
            int thid = (blockIdx.x * blockDim.x) + threadIdx.x;
            if (thid >= n || thid >= d) return;

            int thidx2 = thid << 1;
            int ai = offset * (thidx2 + 1) - 1;
            int bi = ai + offset;

            data[bi] += data[ai];
        }

        __global__ void kernGEM3WarpReorgWorkEfficientScanDownSweep(int n, int d, int offset, int* data) {
            int thid = (blockIdx.x * blockDim.x) + threadIdx.x;
            if (thid >= n || thid >= d) return;

            int thidx2 = thid << 1;
            int ai = offset * (thidx2 + 1) - 1;
            int bi = ai + offset;

            int t = data[bi];
            data[bi] += data[ai];
            data[ai] = t;
        }

        void warpReorgScan(int n, int* dev_data) {
            // upsweep
            int offset = 1;
            for (int d = n >> 1; d > 0; d >>= 1) {
                kernGEM3WarpReorgWorkEfficientScanUpSweep<<<NUM_BLOCKS(d), THREADS_PER_BLOCK>>>(n, d, offset, dev_data);
                offset <<= 1;
            }
            cudaMemset(dev_data + n - 1, 0, sizeof(int));
            // downsweep
            for (int d = 1; d < n; d <<= 1) {
                offset >>= 1;
                kernGEM3WarpReorgWorkEfficientScanDownSweep<<<NUM_BLOCKS(d), THREADS_PER_BLOCK>>>(n, d, offset, dev_data);
            }
        }

#ifndef FULL
        /**
          * Performs prefix-sum (aka scan) on idata, storing the result into odata.
          */
        void scan(int n, int* odata, const int* idata) {
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
            warpReorgScan(nextPow2_n, dev_data);
            // ----------------------------------
            if (!disableScanTimer) timer().endGpuTimer();
            // dev_data now contains an exclusive scan
            cudaMemcpy(odata, dev_data, n * sizeof(int), cudaMemcpyDeviceToHost);
            checkCUDAError("cudaMemcpy dev_data failed!");
            cudaFree(dev_data);
        }
#endif // !FULL
#ifdef FULL

#define ELEMENTS_PER_TILE (2 * BLOCK_SIZE)
#define SHARED_MEM_SIZE (ELEMENTS_PER_TILE * sizeof(int))
#define NUM_BANKS 32
#define LOG_NUM_BANKS 5
#define CONFLICT_FREE_OFFSET(n) ((n) >> NUM_BANKS + (n) >> (2 * LOG_NUM_BANKS))

        __global__ void kernGEM3FullWorkEfficientFixedSizeScan(int* odata, int* idata, int* incr) {
            extern __shared__ int temp[];

            int n = ELEMENTS_PER_TILE; // fixed size scan

            int thid = threadIdx.x;
            int thidx2 = thid << 1;
            int blockOffset = n * blockIdx.x;

            int ai = thid;
            int bi = thid + (n >> 1);
            int bankOffsetA = CONFLICT_FREE_OFFSET(ai);
            int bankOffsetB = CONFLICT_FREE_OFFSET(bi);

            temp[ai + bankOffsetA] = idata[blockOffset + ai];
            temp[bi + bankOffsetB] = idata[blockOffset + bi];

            int offset = 1;

            for (int d = n >> 1; d > 0; d >>= 1) {
                __syncthreads();
                if (thid < d) {
                    int ai = offset * (thidx2 + 1) - 1;
                    int bi = ai + offset;
                    ai += CONFLICT_FREE_OFFSET(ai);
                    bi += CONFLICT_FREE_OFFSET(bi);

                    temp[bi] += temp[ai];
                }
                offset <<= 1;
            }

            __syncthreads();
            if (thid == 0) {
                incr[blockIdx.x] = temp[n - 1 + CONFLICT_FREE_OFFSET(n - 1)];
                temp[n - 1 + CONFLICT_FREE_OFFSET(n - 1)] = 0;
            }

            for (int d = 1; d < n; d <<= 1) {
                offset >>= 1;
                __syncthreads();
                if (thid < d) {
                    int ai = offset * (thidx2 + 1) - 1;
                    int bi = ai + offset;
                    ai += CONFLICT_FREE_OFFSET(ai);
                    bi += CONFLICT_FREE_OFFSET(bi);

                    int t = temp[ai];
                    temp[ai] = temp[bi];
                    temp[bi] += t;
                }
            }
            __syncthreads();

            odata[blockOffset + ai] = temp[ai + bankOffsetA];
            odata[blockOffset + bi] = temp[bi + bankOffsetB];
        }

        __global__ void addIncr(int n, int* dev_data, int* dev_incr) {
            int thid = (blockIdx.x * blockDim.x) + threadIdx.x;
            if (thid >= n) return;
            if (blockIdx.x > 0) dev_data[thid] += dev_incr[blockIdx.x >> 1];
        }

#ifdef RECUR // best for extremely large array, like size >= 2^32
#define THRESHOULD 1 << 16
        // recursive 
        void recurScan(int n, int* dev_odata, int* dev_idata, int* dev_incr) {
            int numTiles = (n + ELEMENTS_PER_TILE - 1) / ELEMENTS_PER_TILE;

            if (numTiles > THRESHOULD) {
                kernGEM3FullWorkEfficientFixedSizeScan<<<numTiles, THREADS_PER_BLOCK, SHARED_MEM_SIZE>>>(dev_odata, dev_idata, dev_incr);
                recurScan(numTiles, dev_incr, dev_incr, dev_incr + numTiles);
                addIncr<<<NUM_BLOCKS(n), THREADS_PER_BLOCK>>>(n, dev_odata, dev_incr);
            } else {
                // only once
                cudaMemcpy(dev_odata, dev_idata, n * sizeof(int), cudaMemcpyDeviceToDevice);
                checkCUDAError("cudaMemcpy dev_odata failed!");
                int nextPow2_n = 1 << ilog2ceil(n);
                warpReorgScan(nextPow2_n, dev_odata);
            }
        }
#endif // RECUR

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int* odata, const int* idata) {
            int numTiles = (n + ELEMENTS_PER_TILE - 1) / ELEMENTS_PER_TILE;
            int nPad = numTiles * ELEMENTS_PER_TILE;
            int* dev_data;
            cudaMalloc((void**)&dev_data, nPad * sizeof(int));
            cudaMemset(dev_data, 0, nPad * sizeof(int));
            cudaMemcpy(dev_data, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            checkCUDAError("cudaMalloc, cudaMemset, cudaMemcpy dev_data failed!");
            int* dev_incr;
            // geometric series guarantees 2 * numTiles * sizeof(int) is enough
            cudaMalloc((void**)&dev_incr, 2 * numTiles * sizeof(int)); 
            checkCUDAError("cudaMalloc dev_incr failed!");
            cudaDeviceSynchronize();
            if (!disableScanTimer) timer().startGpuTimer();
            // ----------------------------------
            // TODO
#ifndef RECUR
            kernGEM3FullWorkEfficientFixedSizeScan<<<numTiles, THREADS_PER_BLOCK, SHARED_MEM_SIZE>>>(dev_data, dev_data, dev_incr);
            warpReorgScan(numTiles, dev_incr);
            addIncr<<<NUM_BLOCKS(n), THREADS_PER_BLOCK>>>(n, dev_data, dev_incr);
#else
            recurScan(n, dev_data, dev_data, dev_incr);
#endif // !RECUR
            // ----------------------------------
            if (!disableScanTimer) timer().endGpuTimer();
            // dev_data now contains an exclusive scan
            cudaMemcpy(odata, dev_data, n * sizeof(int), cudaMemcpyDeviceToHost);
            checkCUDAError("cudaMemcpy dev_data failed!");
            cudaFree(dev_data);
        }
#endif // FULL
#else // naive from lecture
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
            int blockSize = 128;
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
#endif // GEM3

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
            int blockSize = 128;
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

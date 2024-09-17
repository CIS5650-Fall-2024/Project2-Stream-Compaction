#include "common.h"
#include "efficient_share.h"
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/scan.h>

#define blockSize 512
#define itemPerBlock 1024
#define CONFLICT_FREE_OFFSET(n) ((n) >> 5)

namespace StreamCompaction {
    namespace EfficientShare {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        __global__ void kernScan(int n, int* odata, int* sum)
        {
            extern __shared__ int s_odata[];
            int bid = blockIdx.x;
            int tid = threadIdx.x;
            int blockOffset = bid * n;

            // copy to shared buffer
            int ai = tid;
            int bi = tid + (n >> 1);
            int bankOffsetA = CONFLICT_FREE_OFFSET(ai);
            int bankOffsetB = CONFLICT_FREE_OFFSET(bi);
            s_odata[ai + bankOffsetA] = odata[blockOffset + ai];
            s_odata[bi + bankOffsetB] = odata[blockOffset + bi];

            int offset = 1;

            //s_odata[2 * tid] = odata[blockOffset + 2 * tid];
            //s_odata[2 * tid + 1] = odata[blockOffset + 2 * tid + 1];

            // up sweep
            #pragma unroll
            for (int d = itemPerBlock >> 1; d > 0; d >>= 1)
            {
                __syncthreads();
                if (tid < d)
                {
                    int ai = offset * (2 * tid + 1) - 1;
                    int bi = ai + offset;
                    ai += CONFLICT_FREE_OFFSET(ai);
                    bi += CONFLICT_FREE_OFFSET(bi);

                    s_odata[bi] += s_odata[ai];
                    offset <<= 1;
                }
            }

            // set tail to zero
            if (tid == 0)
            {
                sum[bid] = s_odata[n - 1 + CONFLICT_FREE_OFFSET(n - 1)];
                s_odata[n - 1 + CONFLICT_FREE_OFFSET(n - 1)] = 0;
            }

            // down sweep
            #pragma unroll
            for (int d = 1; d < itemPerBlock; d <<= 1)
            {
                __syncthreads();
                if (tid < d)
                {
                    offset >>= 1;
                    int ai = offset * (2 * tid + 1) - 1;
                    int bi = ai + offset;
                    ai += CONFLICT_FREE_OFFSET(ai);
                    bi += CONFLICT_FREE_OFFSET(bi);

                    int t = s_odata[ai];
                    s_odata[ai] = s_odata[bi];
                    s_odata[bi] += t;
                }
            }

            // write back
            __syncthreads();
            odata[blockOffset + ai] = s_odata[ai + bankOffsetA];
            odata[blockOffset + bi] = s_odata[bi + bankOffsetB];
            //odata[blockOffset + 2 * tid] = s_odata[2 * tid];
            //odata[blockOffset + 2 * tid + 1] = s_odata[2 * tid + 1];
        }

        __global__ void kernScanSmall(int n, int* odata)
        {
            extern __shared__ int s_odata[];
            int tid = threadIdx.x;

            // copy to shared buffer
            int ai = tid;
            int bi = tid + (n >> 1);
            int bankOffsetA = CONFLICT_FREE_OFFSET(ai);
            int bankOffsetB = CONFLICT_FREE_OFFSET(bi);

            if (tid < n)
            {
                s_odata[ai + bankOffsetA] = odata[ai];
                s_odata[bi + bankOffsetB] = odata[bi];
            }
            else
            {
                s_odata[ai + bankOffsetA] = 0;
                s_odata[bi + bankOffsetB] = 0;
            }
            

            int offset = 1;

            // up sweep
            for (int d = n >> 1; d > 0; d >>= 1)
            {
                __syncthreads();
                if (tid < d)
                {
                    int ai = offset * (2 * tid + 1) - 1;
                    int bi = ai + offset;
                    ai += CONFLICT_FREE_OFFSET(ai);
                    bi += CONFLICT_FREE_OFFSET(bi);

                    s_odata[bi] += s_odata[ai];
                    offset <<= 1;
                }
            }

            // set tail to zero
            if (tid == 0)
            {
                s_odata[n - 1 + CONFLICT_FREE_OFFSET(n - 1)] = 0;
            }

            // down sweep
            for (int d = 1; d < n; d <<= 1)
            {
                __syncthreads();
                if (tid < d)
                {
                    offset >>= 1;
                    int ai = offset * (2 * tid + 1) - 1;
                    int bi = ai + offset;
                    ai += CONFLICT_FREE_OFFSET(ai);
                    bi += CONFLICT_FREE_OFFSET(bi);

                    int t = s_odata[ai];
                    s_odata[ai] = s_odata[bi];
                    s_odata[bi] += t;
                }
            }

            // write back
            __syncthreads();
            if (tid < n)
            {
                odata[ai] = s_odata[ai + bankOffsetA];
                odata[bi] = s_odata[bi + bankOffsetB];
            }
            
        }

        __global__ void kernAdd(int n, int* odata, const int* incr)
        {
            int bid = blockIdx.x;
            int tid = threadIdx.x;
            int blockOffset = bid * n + tid;

            int stride = n >> 2;
            int base1 = incr[bid];

            bid += gridDim.x;
            int base2 = incr[bid];

            odata[blockOffset] += base1;
            odata[blockOffset + 1 * stride] += base1;
            odata[blockOffset + 2 * stride] += base1;
            odata[blockOffset + 3 * stride] += base1;

            blockOffset = bid * n + tid;
            
            odata[blockOffset] += base2;
            odata[blockOffset + 1 * stride] += base2;
            odata[blockOffset + 2 * stride] += base2;
            odata[blockOffset + 3 * stride] += base2;
        }

        // assume input is already padded
        void scan_dev(int n, int* dev_odata)
        {
            int blockNum = (n + itemPerBlock - 1) / itemPerBlock;
            int pot = nextPowerOfTwo(blockNum);

            timer().pauseGpuTimer();

            int* dev_sum;
            cudaMalloc((void**)&dev_sum, pot * sizeof(int));
            cudaMemset(dev_sum, 0, pot * sizeof(int));

            timer().continueGpuTimer();
            
            kernScan << < blockNum, blockSize, (itemPerBlock + 10) * sizeof(int) >> > (itemPerBlock, dev_odata, dev_sum);

            if (blockNum <= itemPerBlock)
            {
                
                kernScanSmall << < 1, (pot >> 1), pot * sizeof(int) >> > (pot, dev_sum);
            }
            else
            {
                scan_dev(blockNum, dev_sum);
            }

            if (blockNum > 1)
                kernAdd << < (blockNum >> 1), (itemPerBlock >> 2) >> > (itemPerBlock, dev_odata, dev_sum);

            timer().pauseGpuTimer();
            cudaFree(dev_sum);
            timer().continueGpuTimer();

        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int* odata, const int* idata) {
            int blockNum = (n + itemPerBlock - 1) / itemPerBlock;
            int paddedNum = blockNum * itemPerBlock;

            int* dev_obuffer;
            cudaMalloc((void**)&dev_obuffer, paddedNum * sizeof(int));
            cudaMemset(dev_obuffer + n, 0, (paddedNum - n) * sizeof(int));
            cudaMemcpy(dev_obuffer, idata, sizeof(int) * n, cudaMemcpyHostToDevice);

            nvtxRangePushA("Efficient Share");
            timer().startGpuTimer();

            scan_dev(paddedNum, dev_obuffer);

            timer().endGpuTimer();
            nvtxRangePop();

            cudaMemcpy(odata, dev_obuffer, sizeof(int) * n, cudaMemcpyDeviceToHost);
            cudaFree(dev_obuffer);
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

            int blockNum = (n + itemPerBlock - 1) / itemPerBlock;
            int paddedNum = blockNum * itemPerBlock;

            int* dev_idata;
            cudaMalloc((void**)&dev_idata, paddedNum * sizeof(int));
            cudaMemset(dev_idata, 0, paddedNum * sizeof(int));
            cudaMemcpy(dev_idata, idata, sizeof(int) * n, cudaMemcpyHostToDevice);

            int* dev_bools;
            cudaMalloc((void**)&dev_bools, paddedNum * sizeof(int));
            cudaMemset(dev_bools, 0, paddedNum * sizeof(int));

            int* dev_odata;
            cudaMalloc((void**)&dev_odata, paddedNum * sizeof(int));
            cudaMemset(dev_odata, 0, paddedNum * sizeof(int));

            timer().startGpuTimer();

            // map to bool
            int gBlockNum = (paddedNum + blockSize - 1) / blockSize;
            Common::kernMapToBoolean << < gBlockNum, blockSize >> > (paddedNum, dev_bools, dev_idata);

            // scan
            scan_dev(paddedNum, dev_bools);

            // scatter
            Common::kernScatter << < gBlockNum, blockSize >> > (paddedNum, dev_odata, dev_idata, nullptr, dev_bools);

            // copy len back to host
            int len = 0;
            cudaMemcpy(&len, dev_bools + paddedNum - 1, sizeof(int), cudaMemcpyDeviceToHost);

            timer().endGpuTimer();

            cudaMemcpy(odata, dev_odata, sizeof(int) * len, cudaMemcpyDeviceToHost);
            cudaFree(dev_odata);
            cudaFree(dev_idata);
            cudaFree(dev_bools);


            return len;
        }
    }
}
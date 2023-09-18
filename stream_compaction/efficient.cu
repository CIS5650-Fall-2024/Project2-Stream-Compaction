#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

static int blockSize = 64;

namespace StreamCompaction {
    namespace Efficient {
        
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */

        __global__ void kernelUpSweep(int n, int offset, int* i_odata) {
            int threadId = threadIdx.x + blockDim.x * blockIdx.x + 1;
            if (threadId > n)return;
            int id = offset * 2 * threadId - 1;
            i_odata[id] = i_odata[id] + i_odata[id - offset];
        }

        __global__ void kernelDownSweep(int n, int offset, int* i_odata) {
            int threadId = threadIdx.x + blockDim.x * blockIdx.x + 1;
            if (threadId > n)return;
            int id = offset * 2 * threadId - 1;
            //change 2
            int prevIdx = id - offset;
            int prevNum = i_odata[prevIdx];
            i_odata[prevIdx] = i_odata[id];
            i_odata[id] += prevNum;
        }

        void devScan(int* dev_data, int layerCnt, int blockSize) {
            int N = 1 << layerCnt;
            int offset = 1;
            int needN = N;
            for (int i = 0;i < layerCnt;++i) {
                needN /= 2;
                dim3 blockPerGrid((needN + blockSize - 1) / blockSize);
                kernelUpSweep << <blockPerGrid, blockSize >> > (needN, offset, dev_data);
                offset *= 2;
            }
            cudaMemset(dev_data + offset - 1,0,sizeof(int));
            for (int i = 0;i < layerCnt;++i) {
                offset /= 2;
                dim3 blockPerGrid((needN + blockSize - 1) / blockSize);
                kernelDownSweep << <blockPerGrid, blockSize >> > (needN, offset, dev_data);
                needN *= 2;
            }
        }

        void scan(int n, int *odata, const int *idata) {
            
            // TODO
            int* dev_data;
            int layerCnt = ilog2ceil(n);
            int N = 1 << layerCnt;
            cudaMalloc((void**)&dev_data, N * sizeof(int));
            checkCUDAError("cudaMalloc dev_data failed!");
            cudaMemcpy(dev_data, idata, sizeof(int) * n, cudaMemcpyHostToDevice);

            timer().startGpuTimer();

            devScan(dev_data, layerCnt, blockSize);

            timer().endGpuTimer();

            //exclusive scan
            cudaMemcpy(odata, dev_data, sizeof(int) * n, cudaMemcpyDeviceToHost);

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
            
            // TODO
            int layerCnt = ilog2ceil(n);
            int N = 1 << layerCnt;
            int* dev_idata;
            int* dev_odata;
            int* dev_bools;
            int* dev_indices;
            cudaMalloc((void**)&dev_idata, N * sizeof(int));
            checkCUDAError("cudaMalloc dev_idata failed!");
            cudaMalloc((void**)&dev_bools, N * sizeof(int));
            checkCUDAError("cudaMalloc dev_bools failed!");
            cudaMalloc((void**)&dev_odata, N * sizeof(int));
            checkCUDAError("cudaMalloc dev_odata failed!");
            cudaMalloc((void**)&dev_indices, N * sizeof(int));
            checkCUDAError("cudaMalloc dev_indices failed!");
            cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            
            
            dim3 fullBlocksPerGrid((N + blockSize - 1) / blockSize);

            timer().startCpuTimer();

            StreamCompaction::Common::kernMapToBoolean <<<fullBlocksPerGrid, blockSize>>>(N, dev_bools, dev_idata);
            
            cudaMemcpy(dev_indices, dev_bools, N * sizeof(int), cudaMemcpyDeviceToDevice);
            devScan(dev_indices, layerCnt, blockSize);
            
            StreamCompaction::Common::kernScatter << <fullBlocksPerGrid, blockSize >> > (n, dev_odata, dev_idata, dev_bools, dev_indices);
            
            timer().endCpuTimer();

            //read GPU
            int ans = 0;
            cudaMemcpy(odata, dev_odata, sizeof(int) * n, cudaMemcpyDeviceToHost);
            cudaMemcpy(&ans, dev_indices + (N - 1), sizeof(int), cudaMemcpyDeviceToHost);
            int lastBool = 0;
            cudaMemcpy(&lastBool, dev_bools + (N - 1), sizeof(int), cudaMemcpyDeviceToHost);
            ans += lastBool;
            
            //free GPU
            cudaFree(dev_idata);
            cudaFree(dev_bools);
            cudaFree(dev_odata);
            cudaFree(dev_indices);
            
            return ans;
        }

        __global__ void kernMapToBoolean(int n, int* bools, const int* idata, int mask, bool recordZero) {
            // TODO
            int id = blockDim.x * blockIdx.x + threadIdx.x;
            if (id >= n)return;
            bools[id] = (idata[id] & mask) == 0 ? recordZero : (!recordZero);
        }

        __global__ void kernSortScatter(int n, int* odata,
            const int* idata, const int* isZeroBools, 
            const int* indices_0,int zeroCnt) {
            // TODO
            int id = blockDim.x * blockIdx.x + threadIdx.x;
            if (id >= n)return;
            if (isZeroBools[id] == 1) {
                int idx = indices_0[id];
                odata[idx] = idata[id];
            }
            else {
                //ones before current id: id - indices_0[id], therefore we remove a scan 
                int idx = id - indices_0[id] + zeroCnt;
                odata[idx] = idata[id];
            }
        }

        void sort(int n, int* odata, const int* idata) {
            
            // TODO
            int layerCnt = ilog2ceil(n);
            int N = 1 << layerCnt;
            int* dev_idata;
            int* dev_odata;
            int* dev_bools;
            //int* dev_indices_1;
            int* dev_indices_0;
            cudaMalloc((void**)&dev_idata, N * sizeof(int));
            checkCUDAError("cudaMalloc dev_idata failed!");
            cudaMalloc((void**)&dev_bools, N * sizeof(int));
            checkCUDAError("cudaMalloc dev_bools failed!");
            cudaMalloc((void**)&dev_odata, N * sizeof(int));
            checkCUDAError("cudaMalloc dev_odata failed!");
            //cudaMalloc((void**)&dev_indices_1, N * sizeof(int));
            //checkCUDAError("cudaMalloc dev_indices failed!");
            cudaMalloc((void**)&dev_indices_0, N * sizeof(int));
            checkCUDAError("cudaMalloc dev_indices failed!");
            cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            cudaMemset(dev_idata + n, INT_MAX, (N - n)*sizeof(int));//to make non-power-of-two right

            dim3 fullBlocksPerGrid((N + blockSize - 1) / blockSize);
            int mask = 1;

            timer().startGpuTimer();

            for (int i = 0;i < 32;++i) {
                //map 0
                kernMapToBoolean << <fullBlocksPerGrid, blockSize >> > (N, dev_bools, dev_idata,mask,true);
                cudaMemcpy(dev_indices_0, dev_bools, N * sizeof(int), cudaMemcpyDeviceToDevice);
                devScan(dev_indices_0, layerCnt, blockSize);
                int zeroCnt = 0;
                int lastBool = 0;
                cudaMemcpy(&zeroCnt, dev_indices_0 + (N - 1), sizeof(int), cudaMemcpyDeviceToHost);
                cudaMemcpy(&lastBool, dev_bools + (N - 1), sizeof(int), cudaMemcpyDeviceToHost);
                zeroCnt += lastBool;

                //map1
                //kernMapToBoolean << <fullBlocksPerGrid, blockSize >> > (N, dev_bools, dev_idata, mask, false);
                //cudaMemcpy(dev_indices_1, dev_bools, N * sizeof(int), cudaMemcpyDeviceToDevice);
                //devScan(dev_indices_1, layerCnt, blockSize);
                kernSortScatter << <fullBlocksPerGrid, blockSize >> > (N,
                    dev_odata, dev_idata, dev_bools, dev_indices_0,zeroCnt);
                mask <<= 1;
                std::swap(dev_odata, dev_idata);
            }

            timer().endGpuTimer();

            //read GPU
            cudaMemcpy(odata, dev_idata, sizeof(int) * n, cudaMemcpyDeviceToHost);

            //free GPU
            cudaFree(dev_idata);
            cudaFree(dev_bools);
            cudaFree(dev_odata);
            cudaFree(dev_indices_0);
            
        }
    }
}

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "common.h"
#include "efficient.h"

namespace StreamCompaction {
    namespace Efficient {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        void __global__ kernUpSweep(int n, int d, int* idata)
        {
            unsigned int index = threadIdx.x + blockIdx.x * blockDim.x;
            unsigned int offset = (unsigned int)(1 << (d + 1));
            if (index >= (unsigned int)(n >> (d + 1))) return;
            idata[(index + 1) * offset - 1] = idata[(index + 1) * offset - 1] + idata[(index + 1) * offset - 1 - (offset >> 1)];
        }

        void __global__ kernDownSweep(int n, int d, int* idata)
        {
            unsigned int index = threadIdx.x + blockIdx.x * blockDim.x;
            unsigned int offset = (unsigned int)(n >> d);
            if (index >= (unsigned int)(1 << d)) return;
            if (d == 0 && index == 0) idata[n - 1] = 0;
            int temp = idata[(index + 1) * offset - 1];
            idata[(index + 1) * offset - 1] += idata[(index + 1) * offset - 1 - (offset >> 1)];
            idata[(index + 1) * offset - 1 - (offset >> 1)] = temp;
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) 
        {
            unsigned int ilogn = ilog2ceil(n);
            unsigned int numPad0 = 1 << ilogn;
            int* dev_odata;
            cudaMalloc((void**)&dev_odata, numPad0 * sizeof(int));
            checkCUDAError("cudaMalloc dev_odata failed!");
            cudaDeviceSynchronize();

            cudaMemset(dev_odata, 0, numPad0 * sizeof(int));
            checkCUDAError("cudaMemset idata to dev_odata failed!");

            cudaMemcpy(dev_odata, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            checkCUDAError("cudaMempcy idata to dev_odata failed!");
            cudaDeviceSynchronize();

            timer().startGpuTimer();
            // TODO
            for (int d = 0; d < ilogn; ++d)
            {
                unsigned int threadNum = (numPad0 >> (d + 1));
                dim3 blockNum((threadNum + BLOCK_SIZE - 1) / BLOCK_SIZE);
                kernUpSweep << < blockNum, BLOCK_SIZE >> > (numPad0, d, dev_odata);
                //for debug
                //cudaMemcpy(odata, dev_odata, n * sizeof(int), cudaMemcpyDeviceToHost);
                //for (int i = 0; i < n; ++i)
                //{
                //    std::cout << odata[i] << ", ";
                //}
                //std::cout << std::endl;
            }
            for (int d = 0; d < ilogn; ++d)
            {
                unsigned int threadNum = (1 << d);
                dim3 blockNum((threadNum + BLOCK_SIZE - 1) / BLOCK_SIZE);
                kernDownSweep << < blockNum, BLOCK_SIZE >> > (numPad0, d, dev_odata);
                //for debug
                //cudaMemcpy(odata, dev_odata, n * sizeof(int), cudaMemcpyDeviceToHost);
                //for (int i = 0; i < n; ++i)
                //{
                //    std::cout << odata[i] << ", ";
                //}
                //std::cout << std::endl;
            }
            timer().endGpuTimer();
            cudaMemcpy(odata, dev_odata, n * sizeof(int), cudaMemcpyDeviceToHost);
            cudaFree(dev_odata);
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
            unsigned int ilogn = ilog2ceil(n);
            unsigned int numPad0 = 1 << ilogn;
            int* dev_odata;
            int* dev_idata;
            int* dev_mapdata;
            int* dev_scan;

            cudaMalloc((void**)&dev_odata, n * sizeof(int));
            checkCUDAError("cudaMalloc dev_odata failed!");

            cudaMalloc((void**)&dev_mapdata, n * sizeof(int));
            checkCUDAError("cudaMalloc dev_mapdata failed!");

            cudaMalloc((void**)&dev_idata, n * sizeof(int));
            checkCUDAError("cudaMalloc dev_idata failed!");

            cudaMalloc((void**)&dev_scan, numPad0 * sizeof(int));
            checkCUDAError("cudaMalloc dev_scan failed!");

            cudaMemset(dev_scan, 0, numPad0 * sizeof(int));
            checkCUDAError("cudaMemset idata to dev_odata failed!");

            cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);

            dim3 blocksPerGrid((n + BLOCK_SIZE - 1) / BLOCK_SIZE);
            unsigned int countNon0;

            timer().startGpuTimer();
            // TODO
            //first get exclusive scan of bool map in dev_odata
            StreamCompaction::Common::kernMapToBoolean << < blocksPerGrid, BLOCK_SIZE >> > (n, dev_mapdata, dev_idata);

            cudaMemcpy(dev_scan, dev_mapdata, n * sizeof(int), cudaMemcpyDeviceToDevice);
            for (int d = 0; d < ilogn; ++d)
            {
                unsigned int threadNum = (numPad0 >> (d + 1));
                dim3 blockNum((threadNum + BLOCK_SIZE - 1) / BLOCK_SIZE);
                kernUpSweep << < blockNum, BLOCK_SIZE >> > (numPad0, d, dev_scan);
            }
            for (int d = 0; d < ilogn; ++d)
            {
                unsigned int threadNum = (1 << d);
                dim3 blockNum((threadNum + BLOCK_SIZE - 1) / BLOCK_SIZE);
                kernDownSweep << < blockNum, BLOCK_SIZE >> > (numPad0, d, dev_scan);
            }

            StreamCompaction::Common::kernScatter << < blocksPerGrid, BLOCK_SIZE >> > (n, dev_odata, dev_idata, dev_mapdata, dev_scan);

            timer().endGpuTimer();
            cudaMemcpy(odata, dev_odata, n * sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(&countNon0, dev_scan + n - 1, sizeof(int), cudaMemcpyDeviceToHost);
            countNon0 += idata[n - 1] ? 1 : 0;

            cudaFree(dev_idata);
            cudaFree(dev_mapdata);
            cudaFree(dev_odata);
            cudaFree(dev_scan);
            return countNon0;
        }
    }
}

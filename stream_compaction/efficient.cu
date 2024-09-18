#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"
#include "device_launch_parameters.h"

#define BLOCKSIZE 128

namespace StreamCompaction {
    namespace Efficient {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }
        __global__ void kernUpSweep(int n, int i, int* data) {
            int idx = threadIdx.x + (blockIdx.x * blockDim.x);
            if (idx < n) {
                int k1 = 1 << i + 1;
                int k2 = 1 << i;
                if ((idx & (k1 - 1)) == 0) {
                    data[idx + k1 - 1] += data[idx + k2 - 1];
                }
            }
        }
        __global__ void kernDownSweep(int n, int i, int* data) {
            int idx = threadIdx.x + (blockIdx.x * blockDim.x);
            if (idx < n) {
                int k1 = 1 << i + 1;
                int k2 = 1 << i;
                if ((idx & (k1 - 1)) == 0) {
                    int t = data[idx + k2 - 1];
                    data[idx + k2 - 1] = data[idx + k1 - 1];
                    data[idx + k1 - 1] += t;
                }
            }
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            int len = 1 << ilog2ceil(n);
            dim3 blockPerGrid((len + BLOCKSIZE - 1) / BLOCKSIZE);
            int* dev_data;
            cudaMalloc((void**)&dev_data, len * sizeof(int));
            checkCUDAErrorFn("cudaMalloc dev_data failed!");
            cudaMemcpy(dev_data, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            checkCUDAErrorFn("cudaMemcpy dev_data failed!");

            timer().startGpuTimer();
            // up-sweep
            for (int i = 0; i <= ilog2ceil(n) - 1; ++i) {
                kernUpSweep << <blockPerGrid, BLOCKSIZE >> > (len, i, dev_data);
            }
            // set last element 0
            cudaMemset(dev_data + len - 1, 0, sizeof(int));
            // down-sweep
            for (int i = ilog2ceil(n) - 1; i >= 0; --i) {
                kernDownSweep << <blockPerGrid, BLOCKSIZE >> > (len, i, dev_data);
            }
            timer().endGpuTimer();
            cudaMemcpy(odata, dev_data, n * sizeof(int), cudaMemcpyDeviceToHost);
            checkCUDAErrorFn("cudaMemcpy dev_dataToodata failed!");
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
            int len = 1 << ilog2ceil(n);
            dim3 blockPerGrid((len + BLOCKSIZE - 1) / BLOCKSIZE);
            int* dev_odata;
            int* dev_idata;
            int* dev_bool;
            int* dev_indices;
            cudaMalloc((void**)&dev_odata, len * sizeof(int));
            checkCUDAErrorFn("cudaMalloc dev_odata failed!");
            cudaMalloc((void**)&dev_idata, len * sizeof(int));
            checkCUDAErrorFn("cudaMalloc dev_idata failed!");
            cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            cudaMalloc((void**)&dev_bool, len * sizeof(int));
            checkCUDAErrorFn("cudaMalloc dev_bool failed!");
            cudaMalloc((void**)&dev_indices, len * sizeof(int));
            checkCUDAErrorFn("cudaMalloc dev_indices failed!");

            
            timer().startGpuTimer();
            StreamCompaction::Common::kernMapToBoolean <<< blockPerGrid, BLOCKSIZE >> > (len, dev_bool, dev_idata);
            checkCUDAErrorFn("kernMapToBoolean failed!");
            cudaMemcpy(dev_indices, dev_bool, len * sizeof(int), cudaMemcpyDeviceToDevice);
            // up-sweep
            for (int i = 0; i <= ilog2ceil(n) - 1; ++i) {
                kernUpSweep << <blockPerGrid, BLOCKSIZE >> > (len, i, dev_indices);
            }
            // set last element 0
            cudaMemset(dev_indices + len - 1, 0, sizeof(int));
            // down-sweep
            for (int i = ilog2ceil(n) - 1; i >= 0; --i) {
                kernDownSweep << <blockPerGrid, BLOCKSIZE >> > (len, i, dev_indices);
            }
            // scatter
            StreamCompaction::Common::kernScatter << <blockPerGrid, BLOCKSIZE >> > (len, dev_odata, dev_idata, dev_bool, dev_indices);
            checkCUDAErrorFn("kernScatter failed!");
            timer().endGpuTimer();
            cudaMemcpy(odata, dev_odata, n * sizeof(int), cudaMemcpyDeviceToHost);
            int num = 0;
            cudaMemcpy(&num, dev_indices + len - 1, sizeof(int), cudaMemcpyDeviceToHost);
            cudaFree(dev_odata);
            cudaFree(dev_idata);
            cudaFree(dev_bool);
            cudaFree(dev_indices);
            return num;
        }
    }
}

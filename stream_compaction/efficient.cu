#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

#define block_size 128

namespace StreamCompaction {
    namespace Efficient {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        __global__ void kernReduceIter(int n, int* idata, int d) {
            int index = threadIdx.x + (blockIdx.x * blockDim.x);
            // n here is not arr size but num threads needed on iter d
            if (index >= n) {
                return;
            }
            //from slides, k = index*offset
            int offset = 1 << (d + 1);
            idata[index * offset + (1 << (d + 1)) - 1] += idata[index * offset + (1 << d) - 1];
        }

        __global__ void kernDownSweepIter(int n, int* idata, int d) {
            int index = threadIdx.x + (blockIdx.x * blockDim.x);
            // n here is not arr size but num threads needed on iter d
            if (index >= n) {
                return;
            }
            //from slides, k = index*offset
            int offset = 1 << (d + 1);
            int tmp = idata[index * offset + (1 << d) - 1];
            idata[index * offset + (1 << d) - 1] = idata[index * offset + (1 << (d + 1)) - 1];
            idata[index * offset + (1 << (d + 1)) - 1] += tmp;
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            dim3 block_dim((n + block_size - 1) / block_size);
            int* dev_idata, *dev_odata;
            int log2n = ilog2ceil(n);
            int arr_size = 1 << log2n;
            cudaMalloc((void**)&dev_idata, arr_size * sizeof(int));
            checkCUDAError("cudaMalloc dev_idata failed!");
            cudaMalloc((void**)&dev_odata, arr_size * sizeof(int));
            checkCUDAError("cudaMalloc dev_odata failed!");

            //memset pow of 2 arrays to 0 to preset padded vals
            cudaMemset(dev_idata, 0, arr_size * sizeof(int));
            checkCUDAError("cudaMemset dev_idata failed!");
            cudaMemset(dev_odata, 0, arr_size * sizeof(int));
            checkCUDAError("cudaMemset dev_odata failed!");
            cudaMemcpy(dev_idata, idata, sizeof(int) * n, cudaMemcpyHostToDevice);
            checkCUDAError("cudaMemcpy idata into dev_idata failed!");

            timer().startGpuTimer();
            //up sweep in place
            for (int d = 0; d < log2n; d++) {
                kernReduceIter << <dim3((arr_size / (1 << (d + 1)) + block_size - 1) / block_size), block_size >> > (arr_size / (1 << (d + 1)), dev_idata, d);
            }
            //down sweep in place
            //set last elem to 0 through cuda(on dev)
            cudaMemset(&dev_idata[arr_size-1], 0, sizeof(int));
            for (int d = log2n - 1; d >= 0; d--) {
                kernDownSweepIter << < dim3((arr_size / (1 << (d + 1)) + block_size - 1) / block_size), block_size >> > (arr_size / (1 << (d + 1)), dev_idata, d);
            }
            //swap so free logic below remains valid
            std::swap(dev_odata, dev_idata);
            timer().endGpuTimer();

            //only copy first n values(exclude tail padded 0s)
            cudaMemcpy(odata, dev_odata, sizeof(int) * n, cudaMemcpyDeviceToHost);
            checkCUDAError("cudaMemcpy dev_odata into odata failed!");
            cudaFree(dev_odata);
            checkCUDAError("free dev_odata failed!");
            cudaFree(dev_idata);
            checkCUDAError("free dev_idata failed!");
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
            dim3 block_dim((n + block_size - 1) / block_size);
            int* dev_idata, *dev_odata, *dev_filter_map;
            int log2n = ilog2ceil(n);
            int arr_size = 1 << log2n;
            cudaMalloc((void**)&dev_idata, n * sizeof(int));
            checkCUDAError("cudaMalloc dev_idata failed!");
            cudaMalloc((void**)&dev_odata, n * sizeof(int));
            checkCUDAError("cudaMalloc dev_odata failed!");
            //padded for scan
            cudaMalloc((void**)&dev_filter_map, arr_size * sizeof(int));
            checkCUDAError("cudaMalloc dev_filter_map failed!");

            //memset pow of 2 arrays to 0 to preset padded vals
            cudaMemset(dev_filter_map, 0, arr_size * sizeof(int));
            checkCUDAError("cudaMemset dev_filter_map failed!");

            //copy input data to device
            cudaMemcpy(dev_idata, idata, sizeof(int) * n, cudaMemcpyHostToDevice);
            checkCUDAError("cudaMemcpy idata into dev_idata failed!");

            timer().startGpuTimer();
            //generate filter map for input in dev_filter_map
            StreamCompaction::Common::kernMapToBoolean << <block_dim, block_size >> > (n, dev_filter_map, dev_idata);
            //copy paste scan to avoid double timer start
            //in place on filter map since can use idata for bools

            //up sweep in place
            for (int d = 0; d < log2n; d++) {
                kernReduceIter << < dim3((arr_size / (1 << (d + 1)) + block_size - 1) / block_size), block_size >> > (arr_size / (1 << (d + 1)), dev_filter_map, d);
            }
            //down sweep in place
            //set last elem to 0 through cuda(on dev)
            cudaMemset(&dev_filter_map[arr_size - 1], 0, sizeof(int));
            for (int d = log2n - 1; d >= 0; d--) {
                kernDownSweepIter << < dim3((arr_size / (1 << (d + 1)) + block_size - 1) / block_size), block_size >> > (arr_size / (1 << (d + 1)), dev_filter_map, d);
            }
            //dev_filter_map contains indices

            //final obj num
            int num_objs;
            cudaMemcpy(&num_objs, &dev_filter_map[n - 1], sizeof(int), cudaMemcpyDeviceToHost);
            //inc num_objs since exc scan
            if (idata[n - 1] != 0) num_objs++;

            //scatter on output array
            StreamCompaction::Common::kernScatter << <block_dim, block_size >> > (n, dev_odata, dev_idata, dev_idata, dev_filter_map);
            timer().endGpuTimer();

            cudaMemcpy(odata, dev_odata, sizeof(int) * num_objs, cudaMemcpyDeviceToHost);
            checkCUDAError("cudaMemcpy dev_odata into odata failed!");
            cudaFree(dev_odata);
            checkCUDAError("free dev_odata failed!");
            cudaFree(dev_idata);
            checkCUDAError("free dev_idata failed!");
            cudaFree(dev_filter_map);
            checkCUDAError("free dev_filter_map failed!");
            return num_objs;
        }
    }
}

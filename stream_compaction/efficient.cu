#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

#include <device_launch_parameters.h>
#include <thread>
#include <thrust/device_ptr.h>

constexpr int blockSize = 128;

namespace StreamCompaction {
    namespace Efficient {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        __global__ void kernReduce(int N, int d, int *data)
        {
            int index = (blockIdx.x * blockDim.x) + threadIdx.x;
            if (index >= N) { return; }

            int k = index * (1 << (d + 1));

            data[k + (1 << (d + 1)) - 1] += data[k + (1 << d) - 1];
        }

        __global__ void kernTraverseBack(int N, int d, int *data)
        {
            int index = (blockIdx.x * blockDim.x) + threadIdx.x;
            if (index >= N) { return; }

            int k = index * (1 << (d + 1));

            int t = data[k + (1 << d) - 1];
            data[k + (1 << d) - 1] = data[k + (1 << (d + 1)) - 1];
            data[k + (1 << (d + 1)) - 1] += t;
        }

        void scan_gpu(int n, int* dev_data)
        {
            thrust::device_ptr<int> dev_thrust_data(dev_data);

            // Up-sweep
            for (int d = 0; d < ilog2ceil(n) - 1; ++d)
            {
                dim3 threadsPerBlock(blockSize);
                dim3 blocksPerGrid(((n >> (d + 1)) + blockSize - 1) / blockSize);
                kernReduce <<<blocksPerGrid, threadsPerBlock>>> ((n >> (d + 1)), d, dev_data);
                checkCUDAError("kernReduce launch failed!");
            }

            // Down-sweep
            dev_thrust_data[n - 1] = 0;

            for (int d = ilog2ceil(n) - 1; d >= 0; --d)
            {
                int numThreads = n >> (d + 1);
                if (numThreads << (d + 1) != n) { numThreads = n >> d; }
                dim3 threadsPerBlock(blockSize);
                dim3 blocksPerGrid((numThreads + blockSize - 1) / blockSize);
                kernTraverseBack <<<blocksPerGrid, threadsPerBlock>>> (numThreads, d, dev_data);
                checkCUDAError("kernTraverseBack launch failed!");
            }
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            int nCeil = 1 << ilog2ceil(n);
            // Up-sweep
            int *dev_data;
            cudaMalloc((void**)&dev_data, nCeil * sizeof(int));
            checkCUDAError("cudaMalloc dev_data failed!");

            cudaMemcpy(dev_data, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            checkCUDAError("cudaMemcpy idata->dev_data failed!");

            timer().startGpuTimer();
            scan_gpu(nCeil, dev_data);
            timer().endGpuTimer();

            cudaMemcpy(odata, dev_data, n * sizeof(int), cudaMemcpyDeviceToHost);
            checkCUDAError("cudaMemcpy dev_data->odata failed!");

            // Clean up
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
            int nCeil = 1 << ilog2ceil(n);
            size_t sizeOfArrays = nCeil * sizeof(int);

            int* dev_bools;
            cudaMalloc((void**)&dev_bools, sizeOfArrays);
            checkCUDAError("cudaMalloc dev_bools failed!");

            int* dev_indices;
            cudaMalloc((void**)&dev_indices, sizeOfArrays);
            checkCUDAError("cudaMalloc dev_indices failed!");

            thrust::device_ptr<int> dev_thrust_indices(dev_indices);
            thrust::device_ptr<int> dev_thrust_bools(dev_bools);

            int* dev_idata;
            cudaMalloc((void**)&dev_idata, sizeOfArrays);
            checkCUDAError("cudaMalloc dev_idata failed!");

            int* dev_odata;
            cudaMalloc((void**)&dev_odata, sizeOfArrays);
            checkCUDAError("cudaMalloc dev_odata failed!");

            cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            checkCUDAError("cudaMemcpy idata->dev_idata failed!");

            dim3 threadsPerBlock(blockSize);
            dim3 blocksPerGrid((n + blockSize - 1) / blockSize);

            timer().startGpuTimer();
            Common::kernMapToBoolean <<<blocksPerGrid, threadsPerBlock>>> (nCeil, dev_bools, dev_idata);
            checkCUDAError("kernMapToBoolean launch failed!");

            cudaMemcpy(dev_indices, dev_bools, sizeOfArrays, cudaMemcpyDeviceToDevice);
            checkCUDAError("cudaMemcpy dev_bools->dev_indices failed!");
            scan_gpu(nCeil, dev_indices);

            int numRemaining = dev_thrust_indices[n - 1] + dev_thrust_bools[n - 1];

            Common::kernScatter <<<blocksPerGrid, threadsPerBlock>>> (nCeil, dev_odata, dev_idata, dev_bools, dev_indices);
            checkCUDAError("kernScatter launch failed!");
            timer().endGpuTimer();

            cudaMemcpy(odata, dev_odata, numRemaining * sizeof(int), cudaMemcpyDeviceToHost);

            return numRemaining;
        }
    }
}

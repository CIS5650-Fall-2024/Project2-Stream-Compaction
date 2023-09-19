#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"
#include "device_launch_parameters.h"

#define BLOCK_SIZE 128

namespace StreamCompaction {
  namespace Efficient {
    using StreamCompaction::Common::PerformanceTimer;
    PerformanceTimer& timer()
    {
      static PerformanceTimer timer;
      return timer;
    }

    // padded_n must be a power of 2
    __global__ void up_sweep(int* data, int d, int num_thds) {
      int thd_idx = threadIdx.x + blockIdx.x * blockDim.x;
      if (thd_idx >= num_thds) {
        // block not full, terminate threads early
        // number of early terminated threads < BLOCK_SIZE always
        return;
      }
      // index of the element in the array that will be updated
      int arr_idx = ((thd_idx + 1) << (d + 1)) - 1;
      // index of the element whose value will be added to data[arr_idx]
      int add_idx = arr_idx - (1 << d);
      // update element
      data[arr_idx] += data[add_idx];
    }

    __global__ void down_sweep(int* data, int d, int num_thds) {
      int thd_idx = threadIdx.x + blockIdx.x * blockDim.x;
      if (thd_idx >= num_thds) {
        // block not full, terminate threads early
        // number of early terminated threads < BLOCK_SIZE always
        return;
      }
      // index of the left cell, which will inherit value from r_idx
      int l_idx = (thd_idx << (d + 1)) + (1 << d) - 1;
      int r_idx = l_idx + (1 << d);
      int tmp = data[l_idx];
      data[l_idx] = data[r_idx];
      data[r_idx] += tmp;
    }

    __global__ void nullify_last_elem(int padded_n, int* data) {
      data[padded_n - 1] = 0;
    }

    /**
      * Performs prefix-sum (aka scan) on idata, storing the result into odata.
      */
    void scan(int n, int *odata, const int *idata) {
      int layer = ilog2ceil(n);
      int padded_n = 1 << layer;
      int num_thds = padded_n;
      int *dev_buffer;
      cudaMalloc((void**)&dev_buffer, padded_n * sizeof(int));
      checkCUDAError("failed to cudaMalloc buffer");
      cudaMemcpy(dev_buffer, idata, n * sizeof(int), cudaMemcpyHostToDevice);
      checkCUDAError("failed to copy idata to buffer");
      timer().startGpuTimer();
      // TODO
      for (int d = 0; d < layer; d++) {
        // update #threads needed
        num_thds >>= 1;
        dim3 gridDim((num_thds + BLOCK_SIZE - 1) / BLOCK_SIZE);
        up_sweep<<<gridDim, BLOCK_SIZE>>>(dev_buffer, d, num_thds);
      }
      nullify_last_elem<<<1, 1>>>(padded_n, dev_buffer);
      for (int d = layer - 1; d >= 0; d--) {
        dim3 gridDim((num_thds + BLOCK_SIZE - 1) / BLOCK_SIZE);
        down_sweep<<<gridDim, BLOCK_SIZE>>>(dev_buffer, d, num_thds);
        num_thds <<= 1;
      }
      timer().endGpuTimer();
      cudaMemcpy(odata, dev_buffer, n * sizeof(int), cudaMemcpyDeviceToHost);
      checkCUDAError("failed to copy buffer to odata");
      cudaFree(dev_buffer);
      checkCUDAError("failed to free dev_buffer");
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
      timer().startGpuTimer();
      // TODO
      timer().endGpuTimer();
      return -1;
    }
  }
}

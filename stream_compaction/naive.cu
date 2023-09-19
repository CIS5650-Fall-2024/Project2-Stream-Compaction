#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"
#include "device_launch_parameters.h"

#define BLOCK_SIZE 128

namespace StreamCompaction {
  namespace Naive {
    using StreamCompaction::Common::PerformanceTimer;
    PerformanceTimer& timer()
    {
      static PerformanceTimer timer;
      return timer;
    }
    // TODO: __global__

    /**
      * Performs prefix-sum (aka scan) on idata, storing the result into odata.
      */
    void scan(int n, int *odata, const int *idata) {
      dim3 gridDim((n + BLOCK_SIZE - 1) / BLOCK_SIZE);
      int *dev_odata, *dev_idata;
      cudaMalloc((void**)& dev_odata, n * sizeof(int));
      checkCUDAError("failed to malloc dev_odata");
      cudaMalloc((void**)& dev_idata, n * sizeof(int));
      checkCUDAError("failed to malloc dev_idata");
      cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);
      checkCUDAError("failed to copy idata to dev_idata");
      timer().startGpuTimer();
      // TODO
      for (int d = 1; d <= ilog2ceil(n); d++) {
        int offset = 1 << (d - 1);
        scan_single_aggregate<<<gridDim, BLOCK_SIZE>>>(n, dev_odata, dev_idata, offset);
        std::swap(dev_odata, dev_idata);
      }
      inclusive_to_exclusive<<<gridDim, BLOCK_SIZE>>>(n, dev_idata, dev_odata);
      timer().endGpuTimer();
      cudaMemcpy(odata, dev_odata, n * sizeof(int), cudaMemcpyDeviceToHost);
      checkCUDAError("failed to copy dev_idata to odata");
      cudaFree(dev_odata);
      cudaFree(dev_idata);
      checkCUDAError("cudaFree failed");
    }

    __global__ void scan_single_aggregate(int n, int* odata, const int* idata, int offset) {
      int idx = threadIdx.x + blockIdx.x * blockDim.x;
      if (idx >= n) {
        return;
      }
      odata[idx] = idx < offset ? idata[idx] : idata[idx] + idata[idx - offset];
    }

    __global__ void inclusive_to_exclusive(int n, int* incl, int* excl) {
      int idx = threadIdx.x + blockIdx.x * blockDim.x;
      if (idx >= n) {
        return;
      }
      excl[idx] = idx == 0 ? 0 : incl[idx - 1];
    }

    __global__ void exclusive_to_inclusive(int n, int* excl, int* incl, int last_num) {
      int idx = threadIdx.x + blockIdx.x * blockDim.x;
      if (idx >= n) {
        return;
      }
      incl[idx] = idx == n - 1 ? excl[n - 1] + last_num : excl[idx + 1];
    }
  }
}

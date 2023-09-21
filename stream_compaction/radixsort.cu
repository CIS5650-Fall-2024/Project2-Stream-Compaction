#include "efficient.h"
#include "radixsort.h"


namespace StreamCompaction {
  namespace RadixSort {
    StreamCompaction::Common::PerformanceTimer& timer() {
      static StreamCompaction::Common::PerformanceTimer timer;
      return timer;
    }

    __global__ void mapToBool(int* out, const int* in, int n, int mask) {
      int k = threadIdx.x + blockDim.x * blockIdx.x;

      if (k >= n) return;

      // set e array value
      if ((in[k] & mask) == 0) {
        out[k] = 1;
      } else {
        out[k] = 0;
      }

      //out[k] = ((in[k] & mask) == 0) ? 1 : 0;
    }

    __global__ void scatter(int n, int* odata,
      const int* idata, const int* bools, const int* indices) {
      int k = threadIdx.x + blockDim.x * blockIdx.x;

      if (k >= n) return;

      // indices is f array, bools is e array (chapter 39)
      int totalFalse = indices[n - 1] + bools[n - 1];

      // here e is the opposite of significant bit b
      int d = bools[k] ? indices[k] : k - indices[k] + totalFalse;
      odata[d] = idata[k];
    }

    void radixsort(int n, int* out, const int* in) {

      const int numBits = 8 * sizeof(int);  // Assuming 32-bit integers

      int* dev_datai, * dev_datao, * dev_bools, * dev_indices;

      cudaMalloc((void**)&dev_datai, n * sizeof(int));
      cudaMalloc((void**)&dev_datao, n * sizeof(int));
      cudaMalloc((void**)&dev_bools, n * sizeof(int));
      cudaMalloc((void**)&dev_indices, n * sizeof(int));

      cudaMemcpy(dev_datai, in, n * sizeof(int), cudaMemcpyHostToDevice);

      int blockSize = 256;  
      int numBlocks = (n + blockSize - 1) / blockSize;

      timer().startGpuTimer();
      for (int d = 0; d < numBits; ++d) {
        int mask = 1 << d;

        mapToBool << <numBlocks, blockSize >> > (dev_bools, dev_datai, n, mask);

        StreamCompaction::Efficient::scan(n, dev_indices, dev_bools);

        scatter << <numBlocks, blockSize >> > (n, dev_datao, dev_datai, dev_bools, dev_indices);

        std::swap(dev_datai, dev_datao);
      }

      timer().endGpuTimer();

      cudaMemcpy(out, dev_datai, n * sizeof(int), cudaMemcpyDeviceToHost);

      cudaFree(dev_datai);
      cudaFree(dev_datao);
      cudaFree(dev_bools);
      cudaFree(dev_indices);
      
    }
  }
}
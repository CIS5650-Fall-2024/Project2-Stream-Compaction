#include <cuda.h>
#include <cuda_runtime.h>
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

        __global__ void upSweep(int n, int d, int* data) {
          int k = threadIdx.x + blockIdx.x * blockDim.x;
          if (k >= n) return;
          int offset = 1 << (d + 1);

          if (k % offset == 0) {
            int left = (k + offset - 1);
            int right = k + (offset >> 1) - 1;
            if (left < n) {
              data[left] += data[right];
            }
          }
        }

        __global__ void downSweep(int n, int d, int* data) {
          int k = threadIdx.x + blockIdx.x * blockDim.x;
          int offset = 1 << (d + 1);

          if (k % offset == 0) {
            int b = k + (offset >> 1) - 1;
            int a = k + offset - 1;

            if (a < n) {
              int temp = data[b];
              data[b] = data[a];
              data[a] += temp;
            }
          }
        }

        __global__ void upSweepV2(int n, int d, int* data) {
          int k = threadIdx.x + blockIdx.x * blockDim.x;
          if (k >= n) return;
          int stride = 1 << (d + 1);

          if (k < (n / stride)) {
            int left = stride * (k + 1) - 1;
            int right = stride * k + (stride >> 1) - 1;
            data[left] += data[right];
          }
        }

        __global__ void downSweepV2(int n, int d, int* data) {
          int k = threadIdx.x + blockIdx.x * blockDim.x;
          int stride = 1 << (d + 1);

          if (k < (n / stride)) {
            // Compute the indices for the left and right elements to be operated on
            int a = stride * (k + 1) - 1;
            int b = stride * k + (stride >> 1) - 1;

            int temp = data[b];
            data[b] = data[a];
            data[a] += temp;
          }
        }


        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            

            int d = ilog2ceil(n) - 1;

            int extendLength = 1 << (d + 1);
            // Allocate device memory
            int* dev_data;
            cudaMalloc((void**)&dev_data, extendLength * sizeof(int));

            // set the cuda memory
            cudaMemset(dev_data, 0, extendLength * sizeof(int));
            cudaMemcpy(dev_data, idata, n * sizeof(int), cudaMemcpyHostToDevice);

            // Set up execution parameters
            int blockSize = 128;
            int gridSize = (extendLength + blockSize - 1) / blockSize;

            timer().startGpuTimer();
            // ------------------------ Version 1.0 -------------------------------------------
            // 
            // 
            // This is my first trival and it is really slow because of wrap divergence and excess global memory access 
            //// Up-sweep
            //for (int i = 0; i <= d; ++i) {
            //  upSweep << <gridSize, blockSize >> > (extendLength, i, dev_data);
            //}

            //// Clear the last element
            //cudaMemset(&dev_data[extendLength - 1], 0, sizeof(int));

            //// Down-sweep
            //for (int i = d; i >= 0; --i) {
            //  downSweep << <gridSize, blockSize >> > (extendLength, i, dev_data);
            //}
            //_______________________Version 1.0 ___________________________________________________


            // ------------------------ Version 2.0 -------------------------------------------
            // 
            // 
            // index optimizing, thus wraps divergance optimizing
            // Up-sweep
            for (int i = 0; i <= d; ++i) {
              upSweepV2 << <gridSize, blockSize >> > (extendLength, i, dev_data);
            }

            // Clear the last element
            cudaMemset(&dev_data[extendLength - 1], 0, sizeof(int));

            // Down-sweep
            for (int i = d; i >= 0; --i) {
              downSweepV2 << <gridSize, blockSize >> > (extendLength, i, dev_data);
            }
            //_______________________Version 2.0 ___________________________________________________
            timer().endGpuTimer();


            // Copy results back to host
            cudaMemcpy(odata, dev_data, n * sizeof(int), cudaMemcpyDeviceToHost);

            // Free device memory
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

            int d = ilog2ceil(n) - 1;

            int extendLength = 1 << (d + 1);
            // Allocate device memory
            int* dev_idata;
            int* dev_odata;
            int* dev_bools;
            int* dev_indices;
            cudaMalloc((void**)&dev_idata, extendLength * sizeof(int));
            cudaMalloc((void**)&dev_odata, extendLength * sizeof(int));
            cudaMalloc((void**)&dev_bools, extendLength * sizeof(int));
            cudaMalloc((void**)&dev_indices, extendLength * sizeof(int));
            
            // Set up execution parameters
            int blockSize = 128;
            int gridSize = (extendLength + blockSize - 1) / blockSize;

            cudaMemset(dev_bools, 0, extendLength);
            cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            cudaMemset(dev_idata + n, 0, (extendLength - n) * sizeof(int));

            timer().startGpuTimer();

            StreamCompaction::Common::kernMapToBoolean << <gridSize, blockSize >> > (n, dev_bools, dev_idata);

            cudaMemcpy(dev_indices, dev_bools, extendLength * sizeof(int), cudaMemcpyDeviceToDevice);
            // Up-sweep
            for (int i = 0; i <= d; ++i) {
              upSweep << <gridSize, blockSize >> > (extendLength, i, dev_indices);
            }

            // Clear the last element
            cudaMemset(&dev_indices[extendLength - 1], 0, sizeof(int));

            // Down-sweep
            for (int i = d; i >= 0; --i) {
              downSweep << <gridSize, blockSize >> > (extendLength, i, dev_indices);
            }

            StreamCompaction::Common::kernScatter << <gridSize, blockSize >> > (n, dev_odata, dev_idata, dev_bools, dev_indices);

            timer().endGpuTimer();

            // Copy results back to host
            cudaMemcpy(odata, dev_odata, n * sizeof(int), cudaMemcpyDeviceToHost);

            cudaFree(dev_idata);
            cudaFree(dev_odata);
            cudaFree(dev_bools);
            cudaFree(dev_indices);

            

            int i;
            for (i = 0; i < n; i++) {
              if (odata[i] == 0) return i;
            }

            if (i == n) return n;
            return -1;
        }
    }
}

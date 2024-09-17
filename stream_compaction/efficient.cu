#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

#include <device_launch_parameters.h>

#define blockSize 128

namespace StreamCompaction {
    namespace Efficient {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        /**
        * Kernel function for up-sweep
        */
        __global__ void kernUpSweep(int n, int d, int* idata) {

            int k = (blockIdx.x * blockDim.x) + threadIdx.x;
            k *= (1 << (d + 1));

            if (k >= n) {
                return;
            } else {
                // from notes: x[k + 2^(d+1) - 1] += x[k + 2^(d) -1]
                idata[k + (1 << (d + 1)) - 1] += idata[k + (1 << d) - 1];
            }
        }

        /**
        * Kernal function for down-sweep 
        */
        __global__ void kernDownSweep(int n, int d, int* data) {
            
            int k = (blockIdx.x * blockDim.x) + threadIdx.x;
            k *= (1 << (d + 1));

            if (k >= n) {
                return;
            }
            else {
                int t = data[k + (1 << d) - 1]; // save left child
                data[k + (1 << d) - 1] = data[k + (1 << (d + 1)) - 1]; // set left child to this node's val
                data[k + (1 << (d + 1)) - 1] += t; // set right child to old left val + this node's val
            }
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {

            int* dev_idata; // one buffer for in-place scan

            int nextPowSpace = 1 << ilog2ceil(n);

            cudaMalloc((void**)&dev_idata, nextPowSpace * sizeof(int));
            checkCUDAError("cudaMalloc dev_idata failed!");

            cudaMemcpy(dev_idata + (nextPowSpace - n), idata, n * sizeof(int), cudaMemcpyHostToDevice);

            timer().startGpuTimer(); // -----------------------------
            
            // up-sweep 
            for (int d = 0; d < ilog2ceil(n); d++) {

                int n_adj = nextPowSpace / (1 << (d + 1));
                dim3 fullBlocksPerGrid((n_adj + blockSize - 1) / blockSize);

                kernUpSweep<<<fullBlocksPerGrid, blockSize>>>(nextPowSpace, d, dev_idata);
                checkCUDAError("kernUpSweep failed!");
            }

            cudaMemset(dev_idata + (nextPowSpace - 1), 0, sizeof(int));

            // down-sweep
            for (int d = ilog2ceil(n) - 1; d >= 0; d--) {

                int n_adj = nextPowSpace / (1 << (d + 1));
                dim3 fullBlocksPerGrid((n_adj + blockSize - 1) / blockSize);

                kernDownSweep<<<fullBlocksPerGrid, blockSize>>>(nextPowSpace, d, dev_idata);
                checkCUDAError("kernDownSweep failed!");
            }

            timer().endGpuTimer(); // --------------------------------

            cudaMemcpy(odata, dev_idata + (nextPowSpace - n), n * sizeof(int), cudaMemcpyDeviceToHost);
            cudaFree(dev_idata);
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

            int* dev_idata;
            int* dev_scanned; 
            int* dev_bools; 

            int nextPowSpace = 1 << ilog2ceil(n);

            cudaMalloc((void**)&dev_idata, nextPowSpace * sizeof(int));
            checkCUDAError("cudaMalloc dev_idata failed!");
            cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);

            cudaMalloc((void**)&dev_bools, nextPowSpace * sizeof(int));
            checkCUDAError("cudaMalloc dev_bools failed!");

            cudaMalloc((void**)&dev_scanned, nextPowSpace * sizeof(int));
            checkCUDAError("cudaMalloc dev_scattered failed!");

            dim3 fullBlocksPerGrid((n + blockSize - 1) / blockSize);

            timer().startGpuTimer(); // --------------
            
            // create temporary binary array
            StreamCompaction::Common::kernMapToBoolean <<<fullBlocksPerGrid, blockSize >>> (n, dev_bools, dev_idata);
            cudaMemcpy(dev_scanned, dev_bools, nextPowSpace * sizeof(int), cudaMemcpyDeviceToDevice);

            // scan (copied and pasted) -------|

            // up-sweep 
            for (int d = 0; d < ilog2ceil(n); d++) {

                int n_adj = nextPowSpace / (1 << (d + 1));
                dim3 fullBlocksPerGrid_adj((n_adj + blockSize - 1) / blockSize);

                kernUpSweep << <fullBlocksPerGrid_adj, blockSize >> > (nextPowSpace, d, dev_scanned);
                checkCUDAError("kernUpSweep failed!");
            }

            cudaMemset(dev_scanned + (nextPowSpace - 1), 0, sizeof(int));

            // down-sweep
            for (int d = ilog2ceil(n) - 1; d >= 0; d--) {

                int n_adj = nextPowSpace / (1 << (d + 1));
                dim3 fullBlocksPerGrid_adj((n_adj + blockSize - 1) / blockSize);

                kernDownSweep <<<fullBlocksPerGrid_adj, blockSize>>> (nextPowSpace, d, dev_scanned);
                checkCUDAError("kernDownSweep failed!");
            }

            // end scan stuff ------------------|

            // scatter
            StreamCompaction::Common::kernScatter <<<fullBlocksPerGrid, blockSize >>> (n, dev_idata, dev_idata, dev_bools, dev_scanned);

            timer().endGpuTimer(); // ----------------

            int count;
            cudaMemcpy(&count, dev_scanned + (n - 1), sizeof(int), cudaMemcpyDeviceToHost);
            count += (idata[n - 1] != 0);

            cudaMemcpy(odata, dev_idata, n * sizeof(int), cudaMemcpyDeviceToHost);
            
            cudaFree(dev_idata);
            cudaFree(dev_bools);
            cudaFree(dev_scanned);

            return count;
        }
    }
}

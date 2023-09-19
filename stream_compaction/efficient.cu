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

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */

        void scanCore(int n, int* dev_odata) {

            dim3 fullBlocksPerGrid((n + blockSize - 1) / blockSize);
            // Reduce
            int offset = 1;
            for (int d = 0; d < ilog2(n); d++) {
                int operation_number = n / (offset * 2);
                dim3 blocksPerGrid((operation_number + (blockSize - 1)) / blockSize);
                //printf("%d\n", blocksPerGrid.x);
                //printf("Cut off unneccessary threads\n");
                if (blocksPerGrid.x == 1) {
                    kernUpSweep << <1, operation_number >> > (n, offset, dev_odata);
                }
                else 
                    kernUpSweep << <blocksPerGrid, blockSize >> > (n, offset, dev_odata);
                //kernUpSweep << <fullBlocksPerGrid, blockSize >> > (n, offset, dev_odata); // 0.31504 for power of two
                checkCUDAError("kernUpSweep failed");
                offset <<= 1;
            }

            // Down sweep
            for (int d = ilog2(n) - 1; d >= 0; d--) {
                offset = (1 << d);
                int operation_number = n / (offset * 2);
                dim3 blocksPerGrid((operation_number + (blockSize - 1)) / blockSize);
                //printf("%d\n", blocksPerGrid.x);
                //printf("Cut off unneccessary threads\n");
                if (blocksPerGrid.x == 1) {
                    kernDownSweep << <1, operation_number >> > (n, offset, dev_odata);
                }
                else
                    kernDownSweep << <blocksPerGrid, blockSize >> > (n, offset, dev_odata);
                //kernDownSweep << <fullBlocksPerGrid, blockSize >> > (n, offset, dev_odata);
                checkCUDAError("kernDownSweep failed");
            }

        }
        void scan(int n, int *odata, const int *idata) {
            int padded_n = (1 << ilog2ceil(n));
            int* dev_odata;
            cudaMalloc(&dev_odata, padded_n * sizeof(int));
            cudaMemcpy(dev_odata, idata, n * sizeof(int), cudaMemcpyKind::cudaMemcpyHostToDevice);
            timer().startGpuTimer();
            scanCore(padded_n, dev_odata);
            timer().endGpuTimer();
            cudaMemcpy(odata, dev_odata, n * sizeof(int), cudaMemcpyKind::cudaMemcpyDeviceToHost);
            cudaFree(dev_odata);
        }

        __global__ void kernUpSweep(int n,int offset, int* odata1) {
            int index = (blockDim.x * blockIdx.x) + threadIdx.x;
            //printf("%d, %d, %d, %d\n",n,  index, offset, index*offset*2);
            int arrIndex = index * (offset * 2);
            if (arrIndex < n) {
                odata1[arrIndex + offset * 2 - 1] += odata1[arrIndex + offset - 1];
                odata1[n-1] = 0;
            }
        }

        __global__ void kernDownSweep(int n, int offset, int* odata1) {
            int index = (blockDim.x * blockIdx.x) + threadIdx.x;
            //printf("%d, %d, %d, %d\n",n,  index, offset, index*offset*2);
            int arrIndex = index * (offset * 2);
            if (arrIndex < n) {
                int t = odata1[arrIndex + offset - 1];
                odata1[arrIndex + offset - 1] = odata1[arrIndex + offset * 2 - 1];
                odata1[arrIndex + offset * 2 - 1] += t;
            }
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
            /*
                bools, indices should only be allocated on device
                odata and idata needs to be copied to device
            */
            int padded_n = (1 << ilog2ceil(n));

            int* dev_bools;
            /* TODO: Check if remaining part is also zero OR DOESN'T MATTER? */
            cudaMalloc(&dev_bools, padded_n * sizeof(int));
            checkCUDAError("cudaMalloc dev_bools failed");

            int* dev_idata;
            cudaMalloc(&dev_idata, n * sizeof(int));
            checkCUDAError("cudaMalloc dev_idata failed");

            cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyKind::cudaMemcpyHostToDevice);
            checkCUDAError("cudaMemcpy from idata to dev_idata failed");

            int* dev_indices;
            cudaMalloc(&dev_indices, padded_n * sizeof(int));
            checkCUDAError("cudaMalloc dev_indices failed");

            int* dev_odata;
            cudaMalloc(&dev_odata, n * sizeof(int));
            checkCUDAError("cudaMalloc dev_odata failed");

            dim3 fullBlocksPerGrid((n + blockSize - 1) / blockSize);
            StreamCompaction::Common::kernMapToBoolean<<<fullBlocksPerGrid, blockSize>>>(n, dev_bools, dev_idata);
            checkCUDAError("kernMapToBoolean failed");
            
            cudaMemcpy(dev_indices, dev_bools, padded_n * sizeof(int), cudaMemcpyKind::cudaMemcpyDeviceToDevice);

            timer().startGpuTimer();
            scanCore(padded_n, dev_indices);
            timer().endGpuTimer();
            StreamCompaction::Common::kernScatter<<<fullBlocksPerGrid, blockSize>>>(n, dev_odata, dev_idata, dev_bools, dev_indices);
            checkCUDAError("kernScatter failed");

            /* Still got problem here! */
            int length, last_element;
            cudaMemcpy(&length, dev_indices + n - 1, sizeof(int), cudaMemcpyKind::cudaMemcpyDeviceToHost);
            cudaMemcpy(&last_element, dev_bools + n-1, sizeof(int), cudaMemcpyKind::cudaMemcpyDeviceToHost);
            checkCUDAError("cudaMemcpy from dev_indices[n-1] to length failed");
            length += last_element; 
            cudaMemcpy(odata, dev_odata, n * sizeof(int), cudaMemcpyKind::cudaMemcpyDeviceToHost);

            cudaFree(dev_idata);
            cudaFree(dev_odata);
            cudaFree(dev_indices);
            cudaFree(dev_bools);
            
            return length;
        }
    }
}

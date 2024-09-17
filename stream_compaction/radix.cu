#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"
#include "efficient.h"

namespace StreamCompaction {
    namespace RadixSort {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer() {
            static PerformanceTimer timer;
            return timer;
        }

        // Get array e = !b, e arary is the array to return wherether current bit is 0 or 1, if 0 return 1, if 1 return 0
        __global__ void kernMapToReverseBoolean(int n, int* bools, const int* idata, int bitPos) {
            int index = threadIdx.x + (blockIdx.x * blockDim.x);
            if (index >= n) return;
            bools[index] = !((idata[index] >> bitPos) & 1);
        }

        // Get t array
        __global__ void kernMaptoTrues(int n, int* falses, int* trues, int* bools) {
			int index = threadIdx.x + (blockIdx.x * blockDim.x);
			if (index >= n) return;
            int totalFalses = falses[n - 1] + bools[n - 1];
            trues[index] = index - falses[index] + totalFalses;
		}

        // Get output array
        __global__ void kernScatter(int n, int* odata, int* idata,int* falses, int* trues, int* bools) {
            int index = threadIdx.x + (blockIdx.x * blockDim.x);
            if (index >= n) return;
            if (bools[index] == 1) {
                odata[falses[index]] = idata[index];
            }
            else {
				odata[trues[index]] = idata[index];
			}
        }

        void radixSort(int n, int* odata, const int* idata, int bitSize) {
            int* dev_odata;
            int* dev_idata;
            int* dev_bools;
            int* dev_falses;
            int* dev_trues;

            // Allocate memory to GPU
            cudaMalloc((void**)&dev_odata, n * sizeof(int));
            checkCUDAError("cudaMalloc dev_odata failed!");
            cudaMalloc((void**)&dev_idata, n * sizeof(int));
            checkCUDAError("cudaMalloc dev_idata failed!");
            cudaMalloc((void**)&dev_bools, n * sizeof(int));
			checkCUDAError("cudaMalloc dev_bools failed!");
            cudaMalloc((void**)&dev_falses, n * sizeof(int));
			checkCUDAError("cudaMalloc dev_falses failed!");
            cudaMalloc((void**)&dev_trues, n * sizeof(int));
			checkCUDAError("cudaMalloc dev_trues failed!");

            cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            checkCUDAError("cudaMemcpy idata failed!");

            int blockSize = 256;
            dim3 fullBlocksPerGrid((n + blockSize - 1) / blockSize);

            //timer().startGpuTimer();
            for (int bit = 0; bit < bitSize; bit++) {
                //Step 1: Map to boolean, if 1 reutrn 0, if 0 return 1
                // e array --> dev_bools
                kernMapToReverseBoolean << <fullBlocksPerGrid, blockSize >> > (n, dev_bools, dev_idata, bit);
                checkCUDAError("kernMapToReverseBoolean failed!");
                cudaDeviceSynchronize();

                //Step 2: Scan
                // f array --> dev_falses
                StreamCompaction::Efficient::scan(n, dev_falses, dev_bools);
                checkCUDAError("scan failed!");
                cudaDeviceSynchronize();

				//Step 3: Map to Trues
                // t array = i - f[i] + totalFalse --> dev_trues
				kernMaptoTrues << <fullBlocksPerGrid, blockSize >> > (n, dev_falses, dev_trues, dev_bools);
				checkCUDAError("kernMaptoTrues failed!");
				cudaDeviceSynchronize();

				//Step 5: Scatter
                //d[i] = e[i] ? t[i] : f[i]
				kernScatter << <fullBlocksPerGrid, blockSize >> > (n, dev_odata, dev_idata, dev_falses, dev_trues, dev_bools);
				checkCUDAError("kernScatter failed!");
				cudaDeviceSynchronize();

                //Swap
				int* temp = dev_idata;
				dev_idata = dev_odata;
				dev_odata = temp;			
			}
            //timer().endGpuTimer();
            cudaMemcpy(odata, dev_idata, n * sizeof(int), cudaMemcpyDeviceToHost);
            checkCUDAError("cudaMemcpy odata failed!");
			cudaFree(dev_odata);
			cudaFree(dev_idata);
			cudaFree(dev_bools);
			cudaFree(dev_falses);
			cudaFree(dev_trues);
            }
        }
    }
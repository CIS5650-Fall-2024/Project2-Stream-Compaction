#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "radix.h"
#include <climits>
#include "efficient.h"
#include <iostream>

namespace StreamCompaction {
    namespace Radix {
        using StreamCompaction::Common::PerformanceTimer;

        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        const int MAX_BLOCK_SIZE = 1024; // keep this as a power of 2

        void sort(int n, int *odata, const int *idata) {
            int numBitsInInt = sizeof(int) * CHAR_BIT;
            int n_padded = pow(2, ilog2ceil(n));

            int* dev_idata;
            cudaMalloc((void**)&dev_idata, n * sizeof(int));
            cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);

            int* dev_odata;
            cudaMalloc((void**)&dev_odata, n * sizeof(int));

            int* bit_mapped_data;
            cudaMalloc((void**)&bit_mapped_data, n * sizeof(int));

            int* scanned_bit_mapped_data;
            cudaMalloc((void**)&scanned_bit_mapped_data, n_padded * sizeof(int));
            cudaMemset(scanned_bit_mapped_data, 0, n_padded * sizeof(int));


            // Calculate the total amount of memory needed for stored_sums (for the scan step)
            int stored_sums_size = 0;
            for (int i = ilog2ceil(n_padded) - ilog2ceil(MAX_BLOCK_SIZE); i >= 1; i -= ilog2ceil(MAX_BLOCK_SIZE)) {
                stored_sums_size += pow(2, i);
            }

            // temp array used to store last entry per block during upsweep. See kernScan and kernIncrement for use info.
            int* stored_sums; 
            cudaMalloc((void**)&stored_sums, std::max(stored_sums_size, 1) * sizeof(int));

            int blockSize = 1024;
            dim3 blocksPerGrid((n + blockSize - 1) / blockSize);

            timer().startGpuTimer();

            for (int i = 0; i < numBitsInInt; ++i) {
                kernMapBits<<<blocksPerGrid, blockSize>>>(n, bit_mapped_data, dev_idata, i);
                cudaDeviceSynchronize();

                // Scan operates in place, so we need to first copy the bit_mapped_data to scanned_bit_mapped_data so we don't lose it
                cudaMemcpy(scanned_bit_mapped_data, bit_mapped_data, n * sizeof(int), cudaMemcpyDeviceToDevice);
                cudaDeviceSynchronize();

                StreamCompaction::Efficient::scan(n_padded, scanned_bit_mapped_data, stored_sums, 0); 
                cudaDeviceSynchronize();

                kernSort<<<blocksPerGrid, blockSize>>>(n, dev_odata, dev_idata, bit_mapped_data, scanned_bit_mapped_data);
                cudaDeviceSynchronize();

                // Swap the pointers so that the output of the current iteration becomes the input of the next iteration
                int* temp = dev_idata;
                dev_idata = dev_odata;
                dev_odata = temp;
            }
            
            timer().endGpuTimer();

            cudaMemcpy(odata, dev_idata, n * sizeof(int), cudaMemcpyDeviceToHost);

            cudaFree(dev_idata);
            cudaFree(dev_odata);
            cudaFree(bit_mapped_data);
            cudaFree(scanned_bit_mapped_data);
            cudaFree(stored_sums);
        }

        __global__ void kernMapBits(int n, int* bit_mapped_data, const int* dev_data, int bitshift) {
            int index = threadIdx.x + (blockIdx.x * blockDim.x);
            if (index >= n) return;

            bit_mapped_data[index] = !((dev_data[index] >> bitshift) & 1);
        }

        __global__ void kernSort(int n, int* dev_odata, const int* dev_idata, const int* bit_mapped_data, const int* scanned_bit_mapped_data) {
            int index = threadIdx.x + (blockIdx.x * blockDim.x);
            if (index >= n) return;

            int totalFalses = scanned_bit_mapped_data[n - 1] + bit_mapped_data[n - 1];
            int trueIndex = index - scanned_bit_mapped_data[index] + totalFalses;
            int finalIndex = bit_mapped_data[index] ? scanned_bit_mapped_data[index] : trueIndex;

            dev_odata[finalIndex] = dev_idata[index];
        }

    }
}
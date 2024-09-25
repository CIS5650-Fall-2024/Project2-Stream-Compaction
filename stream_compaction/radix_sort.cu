#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/scan.h>
#include <thrust/count.h>
#include "common.h"
#include "thrust.h"
#include "radix_sort.h"
#include "efficient.h"
#include <iomanip>
#include <bitset>

//Reference: https://github.com/mark-poscablo/gpu-radix-sort/tree/master
//Reference: https://developer.download.nvidia.com/video/gputechconf/gtc/2020/presentations/s21572-a-faster-radix-sort-implementation.pdf
//Reference: https://gpuopen.com/download/publications/Introduction_to_GPU_Radix_Sort.pdf

#define BLOCK_SIZE 1024
namespace StreamCompaction {
    namespace Radix {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        // Kernel to compute bit mask for each element
        __global__ void kernCheckZeroBit(int n, const int* idata, int bit, int* bitMasks) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < n) {
                bitMasks[idx] = (idata[idx] >> bit) & 1;
            }
        }

        __global__ void kernGenerateIndices(int N, int totalFalses, const int* e, const int* f, int* outIndices)
        {
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i >= N) return;
            outIndices[i] = e[i] ? f[i] : i - f[i] + totalFalses;
        }

        __global__ void kernScatter(int n, int* odata,
            const int* idata, const int* indices) {
            // TODO
            int idx = blockDim.x * blockIdx.x + threadIdx.x;
            if (idx >= n) return;
            odata[indices[idx]] = idata[idx];
        }

        void printBitMaskResults(const int* input, const int* bitMasks, int n, int bit) {
            std::cout << "\nBit mask results for bit " << bit << ":\n";
            std::cout << std::setw(10) << "Input" << " | "
                << std::setw(32) << "Binary" << " | "
                << "Bit Mask\n";
            std::cout << std::string(50, '-') << "\n";

            for (int i = 0; i < n; i++) {
                std::cout << std::setw(10) << input[i] << " | "
                    << std::setw(32) << std::bitset<32>(input[i]) << " | "
                    << std::setw(8) << bitMasks[i] << "\n";
            }
            std::cout << std::endl;
        }

        void printScanResult(const int* scanResult, int size, const char* message) {
            std::cout << "\n--- Scan Result: " << message << " ---\n";
            std::cout << "Index\tValue\n";
            for (int i = 0; i < size; ++i) {
                std::cout << i << "\t" << scanResult[i] << "\n";
            }
            std::cout << "--- End of Scan Result ---\n\n";
        }

        void sort(int n, int* odata, const int* idata) {
           
            // Allocate device memory
            // Allocate device memory
            int* dev_input, * dev_e, * dev_d, * dev_f, * dev_output;
            int* h_f = new int[n];
            int* h_e = new int[n];
            cudaMalloc((void**)&dev_input, n * sizeof(int));
            cudaMalloc((void**)&dev_d, n * sizeof(int));
            cudaMalloc((void**)&dev_output, n * sizeof(int));
            cudaMalloc((void**)&dev_e, n * sizeof(int));
            cudaMalloc((void**)&dev_f, n * sizeof(int));

            cudaMemcpy(dev_input, idata, n * sizeof(int), cudaMemcpyHostToDevice);

            int nblocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
            timer().startGpuTimer();
            // Iterate over each bit
            for (int bit = 0; bit < 31; bit++) {
                // Step 1: Compute bit mask
                kernCheckZeroBit << <nblocks, BLOCK_SIZE >> > (n, dev_input, bit, dev_e);
                //cudaMemcpy(h_e, dev_e, n * sizeof(int), cudaMemcpyDeviceToHost);
                //printBitMaskResults(idata, h_e, n, bit);

                // Step 2: Perform exclusive scan on bitMasks
                StreamCompaction::Efficient::scan(n, dev_f, dev_e);
                //cudaMemcpy(h_f, dev_f, n * sizeof(int), cudaMemcpyDeviceToHost);
                //printScanResult(h_f, n, "After scan operation");

                // Step 3: Count number of zeros
                int f;
                int e;
                cudaMemcpy(&f, dev_f + n - 1, sizeof(int), cudaMemcpyDeviceToHost);
                cudaMemcpy(&e, dev_e + n - 1, sizeof(int), cudaMemcpyDeviceToHost);
                int totalFalses = f + e;

                // Step 4: Calculate d
                kernGenerateIndices << <nblocks, BLOCK_SIZE >> > (n, totalFalses, dev_e, dev_f, dev_d);

                // Step 5: Scatter the data into odata based on bitMasks and prefixSum
                kernScatter << < nblocks, BLOCK_SIZE >> > (n, dev_output, dev_input, dev_d);

                // Step 6: Swap dev_idata and dev_odata for next iteration
                std::swap(dev_input, dev_output);
            }
            timer().endGpuTimer();
            // Copy result back to host
            cudaMemcpy(odata, dev_input, sizeof(int) * n, cudaMemcpyDeviceToHost);

            // Free device memory
            cudaFree(dev_input);
            cudaFree(dev_d);
            cudaFree(dev_output);
            cudaFree(dev_e);
            cudaFree(dev_f);
            delete[]h_f;
            delete[]h_e;

            
        }

    }
}
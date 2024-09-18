#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include "thrust/sort.h"
#include "common.h"
#include "efficient.h"
#include "sort.h"

#define blockSize 128 

namespace StreamCompaction {
    namespace Sort {
        using StreamCompaction::Common::PerformanceTimer;
        int *dev_idata, *dev_bit_extract_neg, *dev_scan_results, *dev_offset_idx, *dev_final_idx, *dev_odata;

        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        __global__ void compute_not_of_bits(int n, int bit_idx, int *odata, const int *idata) {
            int idx = threadIdx.x + (blockIdx.x * blockDim.x);

            if (idx >= n) {
                return;
            }

            // Extract the bit at the given index and negate it
            odata[idx] = ((idata[idx] >> bit_idx) & 1) ^ 1;
        }
        
        __global__ void compute_offset_indices(int n, int totalFalses, int *odata, const int *idata) {
            int idx = threadIdx.x + (blockIdx.x * blockDim.x);

            if (idx >= n) {
                return;
            }

            odata[idx] = idx - idata[idx] + totalFalses;
        }

        __global__ void compute_final_indices(int n, int *odata, const int *b_arr, const int *t_arr, const int *f_arr) {
            int idx = threadIdx.x + (blockIdx.x * blockDim.x);

            if (idx >= n) {
                return;
            }

            odata[idx] = !b_arr[idx] ? t_arr[idx] : f_arr[idx];
        }

        __global__ void scatter(int n, int *odata, const int *idata, const int *indices) {
            int idx = threadIdx.x + (blockIdx.x * blockDim.x);

            if (idx >= n) {
                return;
            }

            odata[indices[idx]] = idata[idx];
        }

        int get_total_falses(int n, const int* dev_scan_results, const int* dev_bit_extract_neg) {
            if (n == 0) {
                return 0;
            }

            // Allocate host memory for just one element (the last one)
            int h_last_scan_result;
            int h_last_bit_extract_neg;

            // Copy only the last element of dev_scan_results and dev_bit_extract_neg
            cudaMemcpy(&h_last_scan_result, &dev_scan_results[n - 1], sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(&h_last_bit_extract_neg, &dev_bit_extract_neg[n - 1], sizeof(int), cudaMemcpyDeviceToHost);

            // Compute totalFalses using only the last elements
            int totalFalses = h_last_scan_result + (h_last_bit_extract_neg == 1 ? 1 : 0);

            return totalFalses;
        }

        void radix_sort(int n, int *odata, const int *idata) {
            cudaMalloc((void**)&dev_idata, n * sizeof(int));
            cudaMalloc((void**)&dev_bit_extract_neg, n * sizeof(int));
            cudaMalloc((void**)&dev_scan_results, n * sizeof(int));
            cudaMalloc((void**)&dev_offset_idx, n * sizeof(int));
            cudaMalloc((void**)&dev_final_idx, n * sizeof(int));
            cudaMalloc((void**)&dev_odata, n * sizeof(int));

            cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            
            // Set up the grid and block sizes
            dim3 fullBlocksPerGrid((n + blockSize - 1) / blockSize);

            // We are assuming 32 bits for each integer
            // Lopping from 0 will get us the least significant bit first
            for (int i = 0; i < 32; i++) {
                // Split function
                compute_not_of_bits << <fullBlocksPerGrid, blockSize >> > (n, i, dev_bit_extract_neg, dev_idata);
                StreamCompaction::Efficient::scan_without_timer(n, dev_scan_results, dev_bit_extract_neg);
                int totalFalses = get_total_falses(n, dev_scan_results, dev_bit_extract_neg);
                compute_offset_indices << <fullBlocksPerGrid, blockSize >> > (n, totalFalses, dev_offset_idx, dev_scan_results);
                compute_final_indices << <fullBlocksPerGrid, blockSize >> > (n, dev_final_idx, dev_bit_extract_neg, dev_offset_idx, dev_scan_results);
                scatter << <fullBlocksPerGrid, blockSize >> > (n, dev_odata, dev_idata, dev_final_idx);
                cudaMemcpy(dev_idata, dev_odata, n * sizeof(int), cudaMemcpyDeviceToDevice);
            }
            
            cudaMemcpy(odata, dev_odata, n * sizeof(int), cudaMemcpyDeviceToHost);

            cudaFree(dev_idata);
            cudaFree(dev_bit_extract_neg);
            cudaFree(dev_scan_results);
            cudaFree(dev_offset_idx);
            cudaFree(dev_final_idx);
            cudaFree(dev_odata);
        }

        /**
         * Performs radix sort on idata, storing the result into odata.
         * This is the reference to the implementation, 
         * as thrust uses radix sort: https://stackoverflow.com/questions/45267748/which-sorting-algorithm-used-in-thrustsort
         */
        void radix_sort_thrust(int n, int *odata, const int *idata) {
            thrust::device_vector<int> d_idata(idata, idata + n);  // Copy data to device
            thrust::device_vector<int> d_odata(n);                 // Allocate space for output on device

            thrust::sort(d_idata.begin(), d_idata.end());

            thrust::copy(d_idata.begin(), d_idata.end(), odata);
        }
    }
}
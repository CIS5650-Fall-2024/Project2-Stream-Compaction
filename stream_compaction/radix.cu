#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "radix.h"

#include <device_launch_parameters.h>

#define blockSize 128

namespace StreamCompaction {
    namespace Radix {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        // populate dev_b with current bit, but reversed
        __global__ void kernIsolateBit(int n, int bit, int* i, int* b) {
            
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= n) {
                return;
            } else {

                int bitVal = (i[idx] >> bit) & 1;
                b[idx] = bitVal ^ 1;
            }
        }

        // Kernel function for up-sweep
        __global__ void kernUpSweep(int n, int d, int* idata) {

            int k = (blockIdx.x * blockDim.x) + threadIdx.x;
            k *= (1 << (d + 1));

            if (k >= n) {
                return;
            }
            else {
                // from notes: x[k + 2^(d+1) - 1] += x[k + 2^(d) -1]
                idata[k + (1 << (d + 1)) - 1] += idata[k + (1 << d) - 1];
            }
        }

        // Kernal function for down-sweep
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

        // Kernal function for getting address for writing true keys
        __global__ void kernCalcAddress(int n, int tf, int* f, int* t) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= n) {
                return;
            }
            else {
                t[idx] = idx - f[idx] + tf; 
            }
        }

        // Kernal function for getting address for writing true keys
        __global__ void kernScatter(int n, int* d, int* b, int* t, int* f) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= n) {
                return;
            }
            else {
                d[idx] = b[idx] ? f[idx] : t[idx];
            }
        }

        // kernel function to reorder original numbers using index d array
        __global__ void kernReorder(int n, int* d, int* i, int* out) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= n) {
                return;
            }
            else {
                int newIdx = d[idx];
                out[newIdx] = i[idx];
            }
        }

        // radix sort
        void sort(int n, int *odata, const int *idata) {
            
            // labels are from parallel radix sort slides
            int* dev_i;
            int* dev_b;
            int* dev_f;
            int* dev_d; 
            int* dev_o;

            int tf; // total falses

            cudaMalloc((void**)&dev_i, n * sizeof(int));
            checkCUDAError("cudaMalloc dev_i failed!");

            cudaMalloc((void**)&dev_b, n * sizeof(int));
            checkCUDAError("cudaMalloc dev_b failed!");

            cudaMalloc((void**)&dev_f, n * sizeof(int));
            checkCUDAError("cudaMalloc dev_f failed!");

            cudaMalloc((void**)&dev_d, n * sizeof(int));
            checkCUDAError("cudaMalloc dev_d failed!");

            cudaMalloc((void**)&dev_o, n * sizeof(int));
            checkCUDAError("cudaMalloc dev_o failed!");

            cudaMemcpy(dev_i, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            dim3 fullBlocksPerGrid((n + blockSize - 1) / blockSize);

            timer().startGpuTimer(); // --------------

            for (int i = 0; i < 32; i++) {

                // isolate current bit to get i array
                kernIsolateBit << <fullBlocksPerGrid, blockSize >> > (n, i, dev_i, dev_b);
                
                int temp;
                cudaMemcpy(&temp, dev_b + (n - 1), sizeof(int), cudaMemcpyDeviceToHost);
                cudaMemcpy(dev_f, dev_b, n * sizeof(int), cudaMemcpyDeviceToDevice);
                
                // get f array by running scan on e 

                // up-sweep 
                for (int d = 0; d < ilog2ceil(n); d++) {

                    int n_adj = n / (1 << (d + 1));
                    dim3 fullBlocksPerGrid_adj((n_adj + blockSize - 1) / blockSize);

                    kernUpSweep << <fullBlocksPerGrid_adj, blockSize >> > (n, d, dev_f);
                    checkCUDAError("kernUpSweep failed!");
                }

                cudaMemset(dev_f + (n - 1), 0, sizeof(int));

                // down-sweep
                for (int d = ilog2ceil(n) - 1; d >= 0; d--) {

                    int n_adj = n / (1 << (d + 1));
                    dim3 fullBlocksPerGrid_adj((n_adj + blockSize - 1) / blockSize);

                    kernDownSweep << <fullBlocksPerGrid_adj, blockSize >> > (n, d, dev_f);
                    checkCUDAError("kernDownSweep failed!");
                }

                // calculate total falses
                cudaMemcpy(&tf, dev_f + (n - 1), sizeof(int), cudaMemcpyDeviceToHost);
                tf += temp;
                
                // t array (formula)
                kernCalcAddress <<<fullBlocksPerGrid, blockSize >>> (n, tf, dev_f, dev_d);
                
                // scatter based on address to get d
                kernScatter<<<fullBlocksPerGrid, blockSize>>> (n, dev_d, dev_b, dev_d, dev_f);
                
                // match indices back
                kernReorder<<<fullBlocksPerGrid, blockSize >> > (n, dev_d, dev_i, dev_o);
            }

            timer().endGpuTimer(); // ----------------

            cudaMemcpy(odata, dev_o, n * sizeof(int), cudaMemcpyDeviceToHost);

            cudaFree(dev_i);
            cudaFree(dev_b);
            cudaFree(dev_f);
            cudaFree(dev_d);
            cudaFree(dev_o);

        }
    }
}

#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"
#include <iostream>

#define blocksize 512

namespace StreamCompaction {
    namespace Efficient {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        __device__ inline bool base_2_mod_is_0(int x, int mod){
            return !((bool)(x & (mod-1)));
        } 

        __global__ void kernEfficientScanUp(int n, int delta, int *data){
            int k = threadIdx.x + blockIdx.x * blockDim.x;
            if(k >= n){
                return;
            }
            
            if(base_2_mod_is_0(k+1, 2 * delta)){
                data[k] += data[k - delta];
            }

        }

        __global__ void kernEfficientScanDown(int n, int delta, int *data){
            int k = threadIdx.x + blockIdx.x * blockDim.x;
            if(k >= n){
                return;
            }

            if(base_2_mod_is_0(k+1, 2 * delta)){
                int prev_ind = k-delta;
                int tmp = data[prev_ind];
                data[prev_ind] = data[k];
                data[k] += tmp;
            }
        }

        void printArray(int arr[], int size) {
            for(int i = 0; i < size; i++) {
                std::cout << arr[i] << " ";
            }
            std::cout << std::endl;
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            int *dev_data;
            int numBlocks = (n + (blocksize - 1)) / blocksize;
            int num_layers = ilog2ceil(n);
            int evenN = 1 << num_layers;
            cudaMalloc((void**)&dev_data, evenN * sizeof(int));
            cudaMemcpy(dev_data, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            if(evenN > n){
                cudaMemset(dev_data + (n), 0, (evenN - n) * sizeof(int));
            }
            std::cout << "Even N" << evenN << std::endl;
            timer().startGpuTimer();
            int delta = 1; 
            //std::cout << "start" << std::endl;
            //cudaMemcpy(odata, dev_data, (n) * sizeof(int), cudaMemcpyDeviceToHost);
            //printArray(odata, n);
            for(int i = 0; i < num_layers - 1; ++i){
                //std::cout << "delta" << delta << std::endl;
                kernEfficientScanUp<<<numBlocks, blocksize>>>(evenN, delta, dev_data);
                delta = delta << 1;
                cudaDeviceSynchronize();
                //cudaMemcpy(odata, dev_data, (n) * sizeof(int), cudaMemcpyDeviceToHost);
                //printArray(odata, n);
            }
            cudaMemset(dev_data + (evenN-1), 0, sizeof(int));
            //std::cout << "Done Scan Up" << std::endl;
            //cudaMemcpy(odata, dev_data, (n) * sizeof(int), cudaMemcpyDeviceToHost);
            //printArray(odata, n);
            for(int i = 0; i < num_layers; ++i){
                kernEfficientScanDown<<<numBlocks, blocksize>>>(evenN, delta, dev_data);
                delta = delta >> 1;
                cudaDeviceSynchronize();
                //cudaMemcpy(odata, dev_data, (n) * sizeof(int), cudaMemcpyDeviceToHost);
                //printArray(odata, n);
            }


            timer().endGpuTimer();
            cudaMemcpy(odata, dev_data, (n) * sizeof(int), cudaMemcpyDeviceToHost);
            cudaDeviceSynchronize();
            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess) {
                std::cout << "CUDA Error: " << cudaGetErrorString(err) << std::endl;
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

            int *dev_data_bool;
            int *dev_data_indices;
            int *dev_idata;
            int *dev_odata;
            int numBlocks = (n + (blocksize - 1)) / blocksize;
            int num_layers = ilog2ceil(n);
            int evenN = 1 << num_layers;
            int bool_sum;

            /*
            std::cout << "Inp Arr [ ";
            for(int i = 0; i < n; ++i){
                std::cout << idata[i] << ", ";
            }
            std::cout << " ]" << std::endl;
            */

            cudaMalloc((void**)&dev_idata, n * sizeof(int));
            cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);

            cudaMalloc((void**)&dev_odata, n * sizeof(int));

            cudaMalloc((void**)&dev_data_indices, evenN * sizeof(int));
            if(evenN > n){
                cudaMemset(dev_data_indices + (n), 0, (evenN - n) * sizeof(int));
            }

            cudaMalloc((void**)&dev_data_bool, n * sizeof(int));

            /*std::cout << "I Data" << std::endl;
            cudaMemcpy(odata, dev_idata, (n) * sizeof(int), cudaMemcpyDeviceToHost);
            printArray(odata, n);*/


            timer().startGpuTimer();

            Common::kernMapToBoolean<<<numBlocks, blocksize>>>(n, dev_data_bool, dev_idata);
            /*std::cout << "Bool" << std::endl;
            cudaMemcpy(odata, dev_data_bool, (n) * sizeof(int), cudaMemcpyDeviceToHost);
            printArray(odata, n);
            */

            cudaMemcpy(dev_data_indices, dev_data_bool, n * sizeof(int), cudaMemcpyDeviceToDevice);

            int delta = 1; 
            for(int i = 0; i < num_layers; ++i){
                kernEfficientScanUp<<<numBlocks, blocksize>>>(evenN, delta, dev_data_indices);
                delta = delta << 1;
                cudaDeviceSynchronize();
            }
            /*std::cout << "Sum Inds" << std::endl;
            cudaMemcpy(odata, dev_data_indices, (evenN) * sizeof(int), cudaMemcpyDeviceToHost);
            printArray(odata, evenN);
            */
            delta = delta >> 1;
            cudaMemcpy(&bool_sum, &dev_data_indices[evenN - 1], sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemset(dev_data_indices + (evenN-1), 0, sizeof(int));
            for(int i = 0; i < num_layers; ++i){
                kernEfficientScanDown<<<numBlocks, blocksize>>>(evenN, delta, dev_data_indices);
                delta = delta >> 1;
                cudaDeviceSynchronize();
            }

            /*std::cout << "Final Inds" << std::endl;
            cudaMemcpy(odata, dev_data_indices, (evenN) * sizeof(int), cudaMemcpyDeviceToHost);
            printArray(odata, evenN);
            */

            Common::kernScatter<<<numBlocks, blocksize>>>(n, dev_odata, dev_idata, dev_data_bool, dev_data_indices);
            timer().endGpuTimer();
            
            cudaMemcpy(odata, dev_odata, (bool_sum) * sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(odata, dev_odata, (n) * sizeof(int), cudaMemcpyDeviceToHost);
            /*std::cout << "compacted with total " << bool_sum << std::endl;
            printArray(odata, bool_sum);
            */
            cudaDeviceSynchronize();
            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess) {
                std::cout << "CUDA Error: " << cudaGetErrorString(err) << std::endl;
            }

            return bool_sum;
        }
    }
}

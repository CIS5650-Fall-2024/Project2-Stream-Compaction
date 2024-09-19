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
        int block_size = 128;

        __global__ void Upsweep_kernel(int n, int* data, int d) {
            int k = (blockIdx.x * blockDim.x) + threadIdx.x;
            int step = 1 << (d + 1);
            int index = k * step + (step - 1);

            if (index < n) {
                int offset = 1 << d;
                data[index] += data[index - offset];
                
            }

        }
        __global__ void Downsweep_kernel(int n, int* data, int d)
        {
            int k = (blockIdx.x * blockDim.x) + threadIdx.x;
            int step = 1 << (d + 1);
            int index = k * step + (step - 1);

            if (index < n) {
                int offset = 1 << d;
                int t = data[index - offset];
                data[index - offset] = data[index];
                data[index] += t;
            }
        }



        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            int d = ilog2ceil(n);
            int new_n = 1 << d;

            int* dev_data;
            cudaMalloc((void**)&dev_data, new_n * sizeof(int));

            cudaMemset(dev_data, 0, new_n * sizeof(int));
            cudaMemcpy(dev_data, idata, n * sizeof(int), cudaMemcpyHostToDevice);


            timer().startGpuTimer();
            // TODO
            //Upsweep
            for (int h = 0; h < ilog2ceil(new_n); h++) {
                int step = 1 << (h + 1);
                int threads = new_n / step;
                dim3 fullBlocksPerGrid((threads + block_size - 1) / block_size);
                Upsweep_kernel << <fullBlocksPerGrid, block_size >> > (new_n, dev_data, h);
                
            }

            cudaMemset(&dev_data[new_n - 1], 0, sizeof(int));

            for (int d = ilog2ceil(new_n) - 1; d >= 0; d--) {
                int step = 1 << (d + 1);
                int threads = new_n / step;
                dim3 fullBlocksPerGrid((threads + block_size - 1) / block_size);
                Downsweep_kernel << <fullBlocksPerGrid, block_size >> > (new_n, dev_data, d);
                

            }


            timer().endGpuTimer();

            cudaMemcpy(odata, dev_data, n * sizeof(int), cudaMemcpyDeviceToHost);

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
            

            int* dev_idata;
            int* dev_odata;
            int* dev_bools;
            int* dev_indices;

            cudaMalloc((void**)&dev_idata, n * sizeof(int));
            cudaMalloc((void**)&dev_bools, n * sizeof(int));
            cudaMalloc((void**)&dev_indices, n * sizeof(int));
            cudaMalloc((void**)&dev_odata, n * sizeof(int));

            
            cudaMemcpy(dev_idata, idata, n*sizeof(int), cudaMemcpyHostToDevice);
            dim3 fullBlocksPerGrid((n + block_size - 1) / block_size);

            


            //timer().startGpuTimer();
            // TODO
            StreamCompaction::Common::kernMapToBoolean << <fullBlocksPerGrid, block_size >> > (n, dev_bools, dev_idata);
            cudaDeviceSynchronize();

            scan(n, dev_indices, dev_bools);
            cudaDeviceSynchronize();

            StreamCompaction::Common::kernScatter << <fullBlocksPerGrid, block_size >> > (n, dev_odata, dev_idata, dev_bools, dev_indices);
            cudaDeviceSynchronize();

            

            
            //timer().endGpuTimer();


            
            int last_bool;
            int last_index;
            cudaMemcpy(&last_bool, dev_bools + n - 1, sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(&last_index, dev_indices + n - 1, sizeof(int), cudaMemcpyDeviceToHost);

            

            cudaMemcpy(odata, dev_odata, (last_index + last_bool)* sizeof(int), cudaMemcpyDeviceToHost);

            cudaFree(dev_idata);
            cudaFree(dev_bools);
            cudaFree(dev_indices);
            cudaFree(dev_odata);
            return last_index + last_bool;
        }
    }
}

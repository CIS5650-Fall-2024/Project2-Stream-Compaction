#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

#define blockSize 128
namespace StreamCompaction {
    namespace Efficient {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }
        __global__ void upSweep(int n, int base, int* idata) {
            int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
            if (idx >= n) {
                return;
            }
            int k = idx * (1 << base + 1);
            if (k >= n) {
                return;
            }

            idata[k + (1 << base + 1) - 1] += idata[k + (1 << base) - 1];
        }
        // referemce to book page algorithm 4
        __global__ void downSweep(int n, int base, int* idata) {
            int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
            if (idx >= n) {
                return;
            }
            int k = idx * (1 << base + 1);
            if (k >= n) {
                return;
            }

            int t = idata[k + (1 << base) - 1];
            idata[k + (1 << base) - 1] = idata[k + (1 << base + 1) - 1];
            idata[k + (1 << base + 1) - 1] += t;

        }

        void processScan(int n, int ending, int* gpu_idata) {
            

            for (int i = 0; i < ilog2ceil(n); i++) {

                dim3 fullBlocksPerGrid((ending / (1 << (i + 1)) + blockSize - 1) / blockSize);
                upSweep <<<fullBlocksPerGrid, blockSize >>> (n, i, gpu_idata);
            }
            cudaMemset(&gpu_idata[ending - 1], 0, sizeof(int));
            checkCUDAError("error in loop 0");
        
            for (int i = ilog2ceil(n) -1; i>=0; i--) {

                dim3 fullBlocksPerGrid((ending / (1 << (i + 1)) + blockSize - 1) / blockSize);
                downSweep <<<fullBlocksPerGrid, blockSize >>> (n, i, gpu_idata);


            }
            checkCUDAError("error in loop 0111");
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            
            // TODO
            int* gpu_odataa;
            int* gpu_idataa;
            int ending = 1 << ilog2ceil(n);

            cudaMalloc((void**)&gpu_odataa, ending * sizeof(int));
            cudaMalloc((void**)&gpu_idataa, ending * sizeof(int));
            checkCUDAError("memory error 0101!!!!!");
            cudaMemset(gpu_odataa, 0, ending * sizeof(int));
            checkCUDAError("memory error 0102!!!!!");
            cudaMemset(gpu_idataa, 0, ending * sizeof(int));
            checkCUDAError("memory error 0103!!!!!");
            cudaMemcpy(gpu_idataa, idata, sizeof(int) * n, cudaMemcpyHostToDevice);

            checkCUDAError("memory error 01!!!!!");


            timer().startGpuTimer();
            processScan(n, ending, gpu_idataa);
            timer().endGpuTimer();


            checkCUDAError("error in loop final process!!!!!");
            int* temp = gpu_odataa;
            gpu_odataa = gpu_idataa;
            gpu_idataa = temp;

            cudaMemcpy(odata, gpu_odataa, sizeof(int) * n, cudaMemcpyDeviceToHost);

            checkCUDAError("memory error 02!!!!!");

            cudaFree(gpu_odataa);
            cudaFree(gpu_idataa);

            
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
            //timer().startGpuTimer();



            dim3 fullBlocksPerGrid((n + blockSize - 1) / blockSize);
            //dim3 numBlocks((n - 1 + blockSize - 1) / blockSize);

            int* gpu_odata;
            int* gpu_idata;

            int ending = 1 << ilog2ceil(n);
            int* gpu_bool;
            int* gup_sum;


            cudaMalloc((void**)&gpu_odata, n * sizeof(int));
            cudaMalloc((void**)&gpu_idata, n * sizeof(int));
            cudaMalloc((void**)&gpu_bool, n * sizeof(int));
            cudaMalloc((void**)&gup_sum, n * sizeof(int));

            cudaMemcpy(gpu_idata, idata, sizeof(int) * n, cudaMemcpyHostToDevice);

            checkCUDAError("memory error 01!!!!!");

            Common::kernMapToBoolean << <fullBlocksPerGrid, blockSize >> > (n, gpu_bool, gpu_idata);

            scan(n, gup_sum, gpu_bool);

            checkCUDAError("memory error 02!!!!!");

            Common::kernScatter << <fullBlocksPerGrid, blockSize >> > (n, gpu_odata, gpu_idata, gpu_bool, gup_sum);




            // TODO
            //timer().endGpuTimer();

            int counter = -1;

            
            cudaMemcpy(odata, gpu_odata, sizeof(int) * n, cudaMemcpyDeviceToHost);

            cudaMemcpy(&counter, &gup_sum[n-1], sizeof(int), cudaMemcpyDeviceToHost);
            if (idata[n - 1] != 0) {
                counter += 1;
            }


            checkCUDAError("memory error 023!!!!!");
            cudaFree(gpu_odata);
            cudaFree(gpu_idata);

            cudaFree(gpu_bool);
            cudaFree(gup_sum);

           

            return counter;
        }
    }
}

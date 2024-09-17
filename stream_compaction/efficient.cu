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

        __global__ void scan_global(int n, int *odata, int *idata)
        {
            int thid = threadIdx.x + (blockIdx.x * blockDim.x);
            //load data in the global memory
           
            for(int offset = 1; offset < n; offset*=2)
            {                
                int ai = offset * (2*thid + 1) - 1;
                int bi = offset * (2*thid + 2) - 1;
           
                if (ai < n && bi < n)
                {

                    idata[bi] += idata[ai];                           
                }
                if(thid == 0)
                    idata[n-1] =0;
                __syncthreads();

            }

            for (int offset = (n/2); offset >= 1; offset /= 2) // traverse down tree & build scan 
            {

               int ai = offset * (2 * thid + 1) - 1;
               int bi = offset * (2 * thid + 2) - 1;
               if ((ai < n) && (bi < n)) {
                   float t = idata[ai];
                   idata[ai] = idata[bi];
                   idata[bi] += t;
                   }
                   
                   __syncthreads();
        
            }
            if (thid < n/2)
                odata[2 * thid] = idata[2 * thid]; // write results to device memory
            __syncthreads();
            if(thid < n/2) 
                odata[2 * thid + 1] = idata[2 * thid + 1];
             __syncthreads();
        }
        
        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            timer().startGpuTimer();
            // TODO
            int *g_odata;
            int *g_idata;
            cudaMalloc((void**)&g_odata,n*sizeof(int));

            cudaMalloc((void**)&g_idata,n*sizeof(int));

            
            
            cudaMemcpy(g_idata,idata,n*sizeof(int),cudaMemcpyHostToDevice);

            int threadsPerBlock = 256;
            int blocksPerGrid = ((n/2) + threadsPerBlock - 1) / threadsPerBlock;
            scan_global << <blocksPerGrid, threadsPerBlock >> > (n, g_odata, g_idata);

            cudaMemcpy(odata,g_odata,sizeof(int)*n,cudaMemcpyDeviceToHost);

            cudaFree(g_idata);
            cudaFree(g_odata);
            timer().endGpuTimer();
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
            timer().startGpuTimer();
            // TODO
            int *g_bools = 0;
            int *bools = 0;
            int *temp_array = (int*)malloc(sizeof(int)*n);
            int* indices;
            int* g_idata;
            int *g_odata;

            int threadsPerBlock = 256;
            int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
            cudaError_t result = cudaMalloc((void**)(&g_bools), n * sizeof(int));
            if (result != cudaSuccess) {
                fprintf(stderr, "Kernel launch 1 failed: %s\n", cudaGetErrorString(result));
                cudaFree(g_bools);
                timer().endGpuTimer();
                return -1;
            }
            result = cudaMalloc((void**)(&g_idata), n * sizeof(int));
            if (result != cudaSuccess) {
                fprintf(stderr, "Kernel launch 2 failed: %s\n", cudaGetErrorString(result));
                cudaFree(g_bools);
                cudaFree(g_idata);
                timer().endGpuTimer();
                return -1;
            }
            cudaMemcpy(g_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            cudaMalloc((void**)&g_odata,sizeof(int)*n);
            StreamCompaction::Common::kernMapToBoolean<<<blocksPerGrid,threadsPerBlock>>>(n, g_bools, g_idata);
            bools = (int*)malloc(n*sizeof(int));
            cudaDeviceSynchronize();
            cudaMemcpy(bools,g_bools,n*sizeof(int),cudaMemcpyDeviceToHost);

            result = cudaMalloc(&indices, n * sizeof(int));
            timer().endGpuTimer();
            scan(n, temp_array, bools);
            timer().startGpuTimer();
            cudaMemcpy(indices, temp_array, n * sizeof(int), cudaMemcpyHostToDevice);

            StreamCompaction::Common::kernScatter<<<blocksPerGrid,threadsPerBlock>>>(n, g_odata,g_idata, g_bools, indices);
            cudaMemcpy(odata,g_odata,n*sizeof(int),cudaMemcpyDeviceToHost);
            cudaFree(bools);
            cudaFree(indices);
            cudaFree(g_odata);
            timer().endGpuTimer();

            return temp_array[n-1];
        }
    }
}

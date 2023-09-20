#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

bool upgrade = true;
namespace StreamCompaction {
    namespace Efficient {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }
        __global__ void kernEffUpSweep(int n, int division, int* idata) {
            int index = threadIdx.x + (blockIdx.x * blockDim.x);
            if (index >= n) {
                return;
            }
            if (((index+1) &(division-1)) == 0) {
                int div = index-(int)(division >> 1);
                idata[index] += idata[div];
            }
        }

        __global__ void kernEffDownSweep(int n, int division, int* idata) {
            int index = threadIdx.x + (blockIdx.x * blockDim.x);
            if (index >= n) {
                return;
            }
            
            if (((index + 1) & (division - 1)) == 0) {
                int div = index-(int)(division >> 1);
                int temp = idata[index];
                idata[index] += idata[div];
                idata[div] = temp;
            }
        }

        __global__ void kernEffUpSweepNew(int n, int division,int iter, int* idata) {
            int index = threadIdx.x + (blockIdx.x * blockDim.x);
            if (index >= n/division) {
                return;
            }
            //int current = (index + 1) * division - 1;
            int current= (int)((index + 1) <<iter)- 1;
            idata[current] += idata[current-(int)(division>>1)];
        }

        __global__ void kernEffDownSweepNew(int n, int division, int iter, int* idata) {
            int index = threadIdx.x + (blockIdx.x * blockDim.x);
            if (index >= n / division) {
                return;
            }
            //int current = (index + 1) * division - 1;
            int current = (int)((index + 1) <<iter) - 1;
            int div = current - (int)(division >> 1);
            int temp = idata[current];
            idata[current] += idata[div];
            idata[div] = temp;
        }
        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        __global__ void kernSMWorkEfficient(int blockIter,int n, int* increments, int* idata) {
            __shared__ int offset;
            offset = (blockIdx.x * blockSize);
            __syncthreads();

            int thisIdx = (int)(threadIdx.x << 1);
            __shared__ int TMshared[blockSize];
            TMshared[thisIdx] = idata[thisIdx + offset];
            TMshared[thisIdx +1] = idata[thisIdx +1 + offset];
            __syncthreads();
            int division = 2;
            for (int i = 1; i < blockIter; i++) {
                if (threadIdx.x < (int)(1 << (blockIter - i))) {
                    int current = (int)((threadIdx.x + 1) << i) - 1;
                    TMshared[current] += TMshared[current - (int)(division >> 1)];
                }
                division = division << 1;
                __syncthreads();
            }
            TMshared[blockSize - 1]=0;
            //__syncthreads();
            for (int i = blockIter; i >= 1; i--) {
                // 1<<(blockIter -i-1)
                if (threadIdx.x < (int)(1 << (blockIter - i))) {
                    int current = (int)((threadIdx.x + 1) << i) - 1;
                    int temp = TMshared[current];
                    TMshared[current] += TMshared[current - (int)(division >> 1)];
                    TMshared[current - (int)(division >> 1)] = temp;
                }
                division = division >> 1;
                __syncthreads();
            }
            
            idata[thisIdx + offset] = TMshared[thisIdx + 1];
            idata[thisIdx +1+ offset] = (thisIdx + 1== blockSize -1) ? TMshared[thisIdx + 1] + idata[thisIdx + 1 + offset] : TMshared[thisIdx + 2];
            __syncthreads();
            increments[blockIdx.x]= idata[blockSize - 1 + offset];

        }

        __global__ void kernSMAddition(int n, int* increments, int* idata) {
            int index = threadIdx.x + (blockIdx.x * blockDim.x);
            if (blockIdx.x >= n) {
                return;
            }
            __shared__ int incre;
            incre= increments[blockIdx.x];
            idata[index] += incre;
        }

        void oldscan(int n, int* odata, const int* idata) {

            // TODO
            
      
                dim3 threadsPerBlock(blockSize);
                int numOfblock = (n + blockSize - 1) / blockSize;
                int* buffer1;

                int iter = ilog2ceil(n);
                int newsize = 1 << iter;
                cudaMalloc((void**)&buffer1, newsize * sizeof(int));
                cudaMemset(buffer1, 0, newsize * sizeof(int));
                cudaMemcpy(buffer1, idata, n * sizeof(int), cudaMemcpyHostToDevice);

                timer().startGpuTimer();
                for (int i = 2; i < newsize; i = i << 1) {
                    kernEffUpSweep << <numOfblock, threadsPerBlock >> > (newsize, i, buffer1);
                }
                cudaMemset(&buffer1[newsize - 1], 0, sizeof(int));
                for (int i = newsize; i >= 2; i = i >> 1) {
                    kernEffDownSweep << <numOfblock, threadsPerBlock >> > (newsize, i, buffer1);
                }
                timer().endGpuTimer();

                cudaMemcpy(odata, buffer1, n * sizeof(int), cudaMemcpyDeviceToHost);
                cudaFree(buffer1);


        }
        void scanupgrade(int n, int* odata, const int* idata) {
     
                dim3 threadsPerBlock(blockSize);

                int* buffer1;

                int iter = ilog2ceil(n);
                int newsize = 1 << iter;
                cudaMalloc((void**)&buffer1, newsize * sizeof(int));
                int numOfblock = (newsize + blockSize - 1) / blockSize;
                cudaMemset(buffer1, 0, newsize * sizeof(int));
                cudaMemcpy(buffer1, idata, n * sizeof(int), cudaMemcpyHostToDevice);
                int idx = 1;
                timer().startGpuTimer();
                for (int i = 2; i < newsize; i = i << 1) {
                    numOfblock = (newsize / i + blockSize - 1) / blockSize;
                    kernEffUpSweepNew << <numOfblock, threadsPerBlock >> > (newsize, i, idx++, buffer1);
                }
                cudaMemset(&buffer1[newsize - 1], 0, sizeof(int));
                for (int i = newsize; i >= 2; i = i >> 1) {
                    numOfblock = (newsize / i + blockSize - 1) / blockSize;
                    kernEffDownSweepNew << <numOfblock, threadsPerBlock >> > (newsize, i, idx--, buffer1);
                }
                timer().endGpuTimer();

                cudaMemcpy(odata, buffer1, n * sizeof(int), cudaMemcpyDeviceToHost);
                cudaFree(buffer1);
                //cudaFree(buffer2);
       
        }

        void scan(int n, int* odata, const int* idata) {
            dim3 threadsPerBlock(blockSize/2);
            dim3 threadsPerBlockl(blockSize);

            int* buffer1;
            int* increments;
            int newnumblock;
            int idx = 1;
            
            int blockIter = ilog2ceil(blockSize);
            int numOfblock = (n + blockSize - 1) / blockSize;
            int newsize = numOfblock * blockSize;
            cudaMalloc((void**)&buffer1, newsize * sizeof(int));
            cudaMalloc((void**)&increments, numOfblock * sizeof(int));

            cudaMemset(buffer1, 0, newsize * sizeof(int));
            //cudaMemset(increments, 0, numOfblock * sizeof(int));
            cudaMemcpy(buffer1, idata, n * sizeof(int), cudaMemcpyHostToDevice);

            numOfblock = (n + blockSize - 1) / blockSize;
            
            timer().startGpuTimer();
            kernSMWorkEfficient << <numOfblock, threadsPerBlock >> > (blockIter, newsize, increments, buffer1);
            
            for (int i = 2; i < numOfblock; i = i << 1) {
                newnumblock = (numOfblock / i + blockSize - 1) / blockSize;
                kernEffUpSweepNew << <newnumblock, threadsPerBlockl >> > (numOfblock, i, idx++, increments);
            }
            cudaMemset(&increments[numOfblock - 1], 0, sizeof(int));
            for (int i = numOfblock; i >= 2; i = i >> 1) {
                newnumblock = (newsize / i + blockSize - 1) / blockSize;
                kernEffDownSweepNew << <newnumblock, threadsPerBlockl >> > (numOfblock, i, idx--, increments);
            }
            kernSMAddition << <numOfblock, threadsPerBlockl >> > (numOfblock, increments, buffer1);
            timer().endGpuTimer();
            
            odata[0] = 0;
            cudaMemcpy(&odata[1], buffer1, (n-1) * sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(&(odata[n]), increments, (2) * sizeof(int), cudaMemcpyDeviceToHost);
            cudaFree(buffer1);
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
            if (upgrade) {
                dim3 threadsPerBlock(blockSize);
                
                int* buffer1;
                int* bools;
                int* indices;

                int iter = ilog2ceil(n);
                int newsize = 1 << iter;
                cudaMalloc((void**)&buffer1, newsize * sizeof(int));
                cudaMalloc((void**)&bools, newsize * sizeof(int));
                cudaMalloc((void**)&indices, newsize * sizeof(int));
                cudaMemset(buffer1, 0, newsize * sizeof(int));
                cudaMemcpy(buffer1, idata, n * sizeof(int), cudaMemcpyHostToDevice);
                int* outbuffer;
                cudaMalloc((void**)&outbuffer, n * sizeof(int));
                int idx = 1;
                int numOfblock = (newsize + blockSize - 1) / blockSize;
                timer().startGpuTimer();
                Common::kernMapToBoolean << <numOfblock, threadsPerBlock >> > (newsize, bools, buffer1);
                cudaMemcpy(indices, bools, newsize * sizeof(int), cudaMemcpyDeviceToDevice);

                for (int i = 2; i < newsize; i = i << 1) {
                    numOfblock = (newsize / i + blockSize - 1) / blockSize;
                    kernEffUpSweepNew << <numOfblock, threadsPerBlock >> > (newsize, i, idx++,indices);
                }
                cudaMemset(&indices[newsize - 1], 0, sizeof(int));
                for (int i = newsize; i >= 2; i = i >> 1) {
                    numOfblock = (newsize / i + blockSize - 1) / blockSize;
                    kernEffDownSweepNew << <numOfblock, threadsPerBlock >> > (newsize, i, idx--, indices);
                }
                numOfblock = (newsize + blockSize - 1) / blockSize;
                //(int n, int *odata,const int* idata, const int* bools, const int* indices)
                Common::kernScatter << <numOfblock, threadsPerBlock >> > (newsize, outbuffer, buffer1, bools, indices);

                timer().endGpuTimer();

                int outputsize;
                cudaMemcpy(&outputsize, &indices[newsize - 1], sizeof(int), cudaMemcpyDeviceToHost);
                cudaMemcpy(odata, outbuffer, outputsize * sizeof(int), cudaMemcpyDeviceToHost);
                cudaFree(buffer1);
                cudaFree(bools);
                cudaFree(indices);
                cudaFree(outbuffer);

                return outputsize;
            }
            else {
                dim3 threadsPerBlock(blockSize);
                int numOfblock = (n + blockSize - 1) / blockSize;
                int* buffer1;
                int* bools;
                int* indices;

                int iter = ilog2ceil(n);
                int newsize = 1<<iter;
                cudaMalloc((void**)&buffer1, newsize * sizeof(int));
                cudaMalloc((void**)&bools, newsize * sizeof(int));
                cudaMalloc((void**)&indices, newsize * sizeof(int));
                cudaMemset(buffer1, 0, newsize * sizeof(int));
                cudaMemcpy(buffer1, idata, n * sizeof(int), cudaMemcpyHostToDevice);
                int* outbuffer;
                cudaMalloc((void**)&outbuffer, n * sizeof(int));

                timer().startGpuTimer();
                Common::kernMapToBoolean << <numOfblock, threadsPerBlock >> > (newsize, bools, buffer1);
                cudaMemcpy(indices, bools, n * sizeof(int), cudaMemcpyDeviceToDevice);

                for (int i = 2; i <newsize; i = i << 1) {
                    kernEffUpSweep << <numOfblock, threadsPerBlock >> > (newsize, i, indices);
                }
                cudaMemset(&indices[newsize - 1], 0, sizeof(int));
                for (int i = newsize; i >= 2; i = i >> 1) {
                    kernEffDownSweep << <numOfblock, threadsPerBlock >> > (newsize, i, indices);
                }
            
                //(int n, int *odata,const int* idata, const int* bools, const int* indices)
                Common::kernScatter << <numOfblock, threadsPerBlock >> > (newsize, outbuffer,buffer1,bools,indices);

                timer().endGpuTimer();

                int outputsize;
                cudaMemcpy(&outputsize, &indices[newsize - 1], sizeof(int), cudaMemcpyDeviceToHost);
                cudaMemcpy(odata, outbuffer, outputsize * sizeof(int), cudaMemcpyDeviceToHost);
                cudaFree(buffer1);
                cudaFree(bools);
                cudaFree(indices);
                cudaFree(outbuffer);
            
                return outputsize;

            }
        }
    }
}

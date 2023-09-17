#include "common.h"

void checkCUDAErrorFn(const char *msg, const char *file, int line) {
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess == err) {
        return;
    }

    fprintf(stderr, "CUDA error");
    if (file) {
        fprintf(stderr, " (%s:%d)", file, line);
    }
    fprintf(stderr, ": %s: %s\n", msg, cudaGetErrorString(err));
    exit(EXIT_FAILURE);
}


namespace StreamCompaction {
    namespace Common {

        /**
         * Maps an array to an array of 0s and 1s for stream compaction. Elements
         * which map to 0 will be removed, and elements which map to 1 will be kept.
         */
        __global__ void kernMapToBoolean(int n, int* bools, const int* idata) {
            // TODO
            int index = blockIdx.x * blockDim.x + threadIdx.x;
            if (index >= n) return;
            bools[index] = !!idata[index];
        }

        /**
         * Performs scatter on an array. That is, for each element in idata,
         * if input[idx] != 0, it copies idata[idx] to odata[indices[idx]].
         */
        __global__ void kernScatter(int N,const int * prefix,
                const int * input, int * output, bool compact) {
            int index = blockIdx.x * blockDim.x + threadIdx.x;
            if (index >= N) return;
            if (!compact||input[index])
            {
                output[prefix[index]] = input[index];
            }
        }

    }
}

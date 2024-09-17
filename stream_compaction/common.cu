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
        __global__ void kernMapToBoolean(int n, int *bools, const int *idata) {

            // compute the thread index
            const unsigned int index {blockIdx.x * blockDim.x + threadIdx.x};

            // avoid execution when the index is out of range
            if (index >= n) {
                return;
            }

            // map non-zero values to one and zeros to zero
            bools[index] = idata[index] ? 1 : 0;
        }

        /**
         * Performs scatter on an array. That is, for each element in idata,
         * if bools[idx] == 1, it copies idata[idx] to odata[indices[idx]].
         */
        __global__ void kernScatter(int n, int *odata,
                const int *idata, const int *bools, const int *indices) {

            // compute the thread index
            const unsigned int index {blockIdx.x * blockDim.x + threadIdx.x};

            // avoid execution when the index is out of range
            if (index >= n) {
                return;
            }

            // process only when the condition is true
            if (bools[index]) {

                // store the output at the desired index
                odata[indices[index]] = idata[index];
            }
        }

    }
}

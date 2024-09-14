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
            int index = threadIdx.x + (blockIdx.x * blockDim.x);
            if (index >= n) return;

            bools[index] = (idata[index] == 0);
        }

        /**
         * Performs scatter on an array. That is, for each element in idata,
         * if bools[idx] == 1, it copies idata[idx] to odata[indices[idx]].
         */
        __global__ void kernScatter(int n, int *odata,
                const int *idata, const int *scannedBools) {
            int threadId = threadIdx.x + (blockIdx.x * blockDim.x);
            if (threadId >= n) return;

            int data = idata[threadId];
            int scan_i = scannedBools[threadId];

            // Special case for last element of idata array
            if (threadId == n - 1 && data) {
                odata[scan_i] = data;
                return;
            }

            int scan_iplusone = scannedBools[threadId + 1];
            if (scan_i != scan_iplusone) {
                odata[scan_i] = data;
            }
        }

    }
}

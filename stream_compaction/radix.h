#pragma once

#include "common.h"

namespace StreamCompaction {
    namespace Radix {
        StreamCompaction::Common::PerformanceTimer& timer();

        void sort(int n, int *odata, const int *idata);
        __global__ void kernMapBits(int n, int *odata, const int *idata, int bitNumber);
        __global__ void kernSort(int n, int *dev_odata, const int* dev_idata, const int *bit_mapped_data, const int *scanned_bit_mapped_data);
    }
}

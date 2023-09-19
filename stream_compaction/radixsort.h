#pragma once

#include "common.h"

namespace StreamCompaction {
  namespace RadixSort {
    StreamCompaction::Common::PerformanceTimer& timer();

    void radixsort(int n, int* out, const int* in);
  }
}
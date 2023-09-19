#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

namespace StreamCompaction {
	namespace RadixSort {
		using StreamCompaction::Common::PerformanceTimer;
		PerformanceTimer& timer()
		{
			static PerformanceTimer timer;
			return timer;
		}

		void sort(int n, int *odata, const int *idata) {
			timer().startGpuTimer();

			timer().endGpuTimer();
		}
	}
}

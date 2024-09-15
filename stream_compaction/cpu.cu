#include <cstdio>
#include "cpu.h"

#include "common.h"

namespace StreamCompaction {
	namespace CPU {
		using StreamCompaction::Common::PerformanceTimer;
		PerformanceTimer& timer()
		{
			static PerformanceTimer timer;
			return timer;
		}

		/**
		 * CPU scan (prefix sum).
		 * For performance analysis, this is supposed to be a simple for loop.
		 * (Optional) For better understanding before starting moving to GPU, you can simulate your GPU scan in this function first.
		 */

		//exlusive scan
		void scan(int n, int* odata, const int* idata) {
			timer().startCpuTimer();
			odata[0] = 0;
			for (int i = 1; i < n; i++) {
				odata[i] = odata[i - 1] + idata[i - 1];
			}
			timer().endCpuTimer();
		}

		/**
		 * CPU stream compaction without using the scan function.
		 *
		 * @returns the number of elements remaining after compaction.
		 */

		//should be exclusive
		int compactWithoutScan(int n, int* odata, const int* idata) {
			timer().startCpuTimer();
			int count = 0; 
			// very naive without scan
			for (int i = 0; i < n; i++) {
				if (idata[i]) {
					odata[count] = idata[i]; 
					count++;
				}
			}
			timer().endCpuTimer();
			return count;
		}

		/**
		 * CPU stream compaction using scan and scatter, like the parallel version.
		 *
		 * @returns the number of elements remaining after compaction.
		 */
		int compactWithScan(int n, int* odata, const int* idata) {
			timer().startCpuTimer();
			//label non-zero elements
			int* nonZero = new int[n];
			for (int i = 0; i < n; i++) {
				nonZero[i] = idata[i] ? 1 : 0;
			}

			//exclusive scan
			//simply copy to prevent double calling timer
			//scan(n, scanResult, nonZero);
			int* scanResult = new int[n];
			scanResult[0] = 0;
			for (int i = 1; i < n; i++) {
				scanResult[i] = scanResult[i - 1] + nonZero[i - 1];
			}

			//scatter
			for (int i = 0; i < n; i++) {
				if (nonZero[i]) {
					odata[scanResult[i]] = idata[i];
				}
			}
			int count = scanResult[n - 1] + nonZero[n - 1];
			delete[] nonZero;
			delete[] scanResult;
			timer().endCpuTimer();
			return count;
		}
	}
}

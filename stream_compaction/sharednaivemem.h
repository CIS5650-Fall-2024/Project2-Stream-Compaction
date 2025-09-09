#pragma once

namespace StreamCompaction {
    namespace SharedNaiveMem {

        StreamCompaction::Common::PerformanceTimer& timer();

        void scanSharedNaive(int n, int* odata, const int* idata);

        int compactNaive(int n, int* odata, const int* idata);

    }
}

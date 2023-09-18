#pragma once

#include "common.h"
#include<vector>

namespace StreamCompaction {
    namespace Efficient {
        class devDataBuffer {
        private:
            int* dev_data;
            int totalSize, size_;
            std::vector<int> sizes;
            std::vector<int> offsets;
        public:
            devDataBuffer(int n, int blockSize, int minSize) :totalSize(0), size_(0) {
                int extendedSize = BLOCKSPERGRID(n, blockSize) * blockSize;
                while (extendedSize > minSize) {
                    if (extendedSize < blockSize) {
                        break;
                    }
                    size_++;
                    sizes.push_back(extendedSize);
                    offsets.push_back(totalSize);
                    totalSize += extendedSize;
                    extendedSize = BLOCKSPERGRID(extendedSize, blockSize);
                }
                cudaMalloc((void**)&dev_data, sizeof(int) * totalSize);
            }
            ~devDataBuffer() {
                cudaFree(dev_data);
            }
            int* operator[](int i) const {
                return dev_data + offsets[i];
            }
            int* data() const {
                return dev_data;
            }
            int size() const {
                return size_;
            }
            int memCnt()const {
                return totalSize;
            }
            int sizeAt(int i) const {
                return sizes[i];
            }

        };

        StreamCompaction::Common::PerformanceTimer& timer();

        void scanInplace(int n, int* dev_data);

        void scan(int n, int* odata, const int* idata);

        void scanShared(int n, int* odata, const int* idata);
        void scanSharedNaive(int n, int* odata, const int* idata);

        int compact(int n, int* odata, const int* idata);
    }
}

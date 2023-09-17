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
            devDataBuffer(int n, int blockSize, int minSize) :totalSize(0), size_(1) {
                int extenedSize = BLOCKPERGRID(n, blockSize) * blockSize;
                while (extenedSize > minSize) {
                    size_++;
                    sizes.push_back(extenedSize);
                    offsets.push_back(totalSize);
                    totalSize += extenedSize;
                    extenedSize = BLOCKPERGRID(extenedSize, blockSize);
                }
                sizes.push_back(extenedSize);
                offsets.push_back(totalSize);
                totalSize += extenedSize;
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

        int compact(int n, int* odata, const int* idata);
    }
}

/**
 * @file      main.cpp
 * @brief     Stream compaction test program
 * @authors   Kai Ninomiya
 * @date      2015
 * @copyright University of Pennsylvania
 */

#include <cstdio>
#include <stream_compaction/cpu.h>
#include <stream_compaction/naive.h>
#include <stream_compaction/efficient.h>
#include <stream_compaction/thrust.h>
#include <stream_compaction/efficient_share.h>
#include "testing_helpers.hpp"

const int POTSIZE = 1 << 25; // feel free to change the size of array
const int NPOT = POTSIZE - 3; // Non-Power-Of-Two
int *a = new int[POTSIZE];
int *b = new int[POTSIZE];
int *c = new int[POTSIZE];

int main(int argc, char* argv[]) {
    // Scan tests

    printf("\n");
    printf("****************\n");
    printf("** SCAN TESTS **\n");
    printf("****************\n");

    genArray(POTSIZE - 1, a, 10);  // Leave a 0 at the end to test that edge case
    a[POTSIZE - 1] = 0;
    printArray(POTSIZE, a, true);


    // initialize b using StreamCompaction::CPU::scan you implement
    // We use b for further comparison. Make sure your StreamCompaction::CPU::scan is correct.
    // At first all cases passed because b && c are all zeroes.
    zeroArray(POTSIZE, b);
    printDesc("cpu scan, power-of-two");
    StreamCompaction::CPU::scan(POTSIZE, b, a);
    printElapsedTime(StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation(), "(std::chrono Measured)");
    printArray(POTSIZE, b, true);

    zeroArray(POTSIZE, c);
    printDesc("cpu scan, non-power-of-two");
    StreamCompaction::CPU::scan(NPOT, c, a);
    printElapsedTime(StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation(), "(std::chrono Measured)");
    printArray(NPOT, b, true);
    printCmpResult(NPOT, b, c);

    /*

    zeroArray(POTSIZE, c);
    printDesc("thrust scan, power-of-two");
    //StreamCompaction::Thrust::scan(POTSIZE, c, a);
    printElapsedTime(StreamCompaction::Thrust::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
    //printArray(POTSIZE, c, true);
    printCmpResult(POTSIZE, b, c);

    zeroArray(POTSIZE, c);
    printDesc("thrust scan, non-power-of-two");
    StreamCompaction::Efficient::scan(NPOT, c, a);
    printElapsedTime(StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
    //printArray(NPOT, c, true);
    printCmpResult(NPOT, b, c);

    zeroArray(POTSIZE, c);
    printDesc("work-efficient scan using shared memory, power-of-two");
    //StreamCompaction::EfficientShare::scan(POTSIZE, c, a);
    printElapsedTime(StreamCompaction::EfficientShare::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
    //printArray(POTSIZE, c, true);
    printCmpResult(POTSIZE, b, c);

    zeroArray(POTSIZE, c);
    printDesc("work-efficient scan using shared memory, non-power-of-two");
    StreamCompaction::EfficientShare::scan(NPOT, c, a);
    printElapsedTime(StreamCompaction::EfficientShare::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
    //printArray(NPOT, c, true);
    printCmpResult(NPOT, b, c);

    zeroArray(POTSIZE, c);
    printDesc("thrust scan, power-of-two");
    StreamCompaction::Thrust::scan(NPOT, c, a);
    printElapsedTime(StreamCompaction::Thrust::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
    //printArray(POTSIZE, c, true);
    printCmpResult(NPOT, b, c);

    */

    

    zeroArray(POTSIZE, c);
    printDesc("naive scan, power-of-two");
    StreamCompaction::Naive::scan(POTSIZE, c, a);
    printElapsedTime(StreamCompaction::Naive::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
    printArray(POTSIZE, c, true);
    printCmpResult(POTSIZE, b, c);

    zeroArray(POTSIZE, c);
    printDesc("naive scan, non-power-of-two");
    StreamCompaction::Naive::scan(NPOT, c, a);
    printElapsedTime(StreamCompaction::Naive::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
    printArray(NPOT, c, true);
    printCmpResult(NPOT, b, c);

    zeroArray(POTSIZE, c);
    printDesc("work-efficient scan, power-of-two");
    StreamCompaction::Efficient::scan(POTSIZE, c, a);
    printElapsedTime(StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
    //printArray(POTSIZE, c, true);
    printCmpResult(POTSIZE, b, c);

    zeroArray(POTSIZE, c);
    printDesc("work-efficient scan, non-power-of-two");
    StreamCompaction::Efficient::scan(NPOT, c, a);
    printElapsedTime(StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
    //printArray(NPOT, c, true);
    printCmpResult(NPOT, b, c);

    zeroArray(POTSIZE, c);
    printDesc("work-efficient scan using shared memory, power-of-two");
    StreamCompaction::EfficientShare::scan(POTSIZE, c, a);
    StreamCompaction::EfficientShare::scan(POTSIZE, c, a);
    printElapsedTime(StreamCompaction::EfficientShare::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
    //printArray(POTSIZE, c, true);
    printCmpResult(POTSIZE, b, c);

    zeroArray(POTSIZE, c);
    printDesc("work-efficient scan using shared memory, non-power-of-two");
    StreamCompaction::EfficientShare::scan(NPOT, c, a);
    printElapsedTime(StreamCompaction::EfficientShare::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
    //printArray(NPOT, c, true);
    printCmpResult(NPOT, b, c);

    zeroArray(POTSIZE, c);
    printDesc("thrust scan, power-of-two");
    StreamCompaction::Thrust::scan(POTSIZE, c, a);
    printElapsedTime(StreamCompaction::Thrust::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
    //printArray(POTSIZE, c, true);
    printCmpResult(POTSIZE, b, c);

    zeroArray(POTSIZE, c);
    printDesc("thrust scan, non-power-of-two");
    StreamCompaction::Thrust::scan(NPOT, c, a);
    printElapsedTime(StreamCompaction::Thrust::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
    //printArray(NPOT, c, true);
    printCmpResult(NPOT, b, c);

    

    

    printf("\n");
    printf("*****************************\n");
    printf("** STREAM COMPACTION TESTS **\n");
    printf("*****************************\n");

    // Compaction tests

    genArray(POTSIZE - 1, a, 4);  // Leave a 0 at the end to test that edge case
    a[POTSIZE - 1] = 0;
    printArray(POTSIZE, a, true);

    int count, expectedCount, expectedNPOT;

    // initialize b using StreamCompaction::CPU::compactWithoutScan you implement
    // We use b for further comparison. Make sure your StreamCompaction::CPU::compactWithoutScan is correct.
    zeroArray(POTSIZE, b);
    printDesc("cpu compact without scan, power-of-two");
    count = StreamCompaction::CPU::compactWithoutScan(POTSIZE, b, a);
    printElapsedTime(StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation(), "(std::chrono Measured)");
    expectedCount = count;
    printArray(count, b, true);
    printCmpLenResult(count, expectedCount, b, b);

    zeroArray(POTSIZE, c);
    printDesc("cpu compact without scan, non-power-of-two");
    count = StreamCompaction::CPU::compactWithoutScan(NPOT, c, a);
    printElapsedTime(StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation(), "(std::chrono Measured)");
    expectedNPOT = count;
    printArray(count, c, true);
    printCmpLenResult(count, expectedNPOT, b, c);

    zeroArray(POTSIZE, c);
    printDesc("cpu compact with scan");
    count = StreamCompaction::CPU::compactWithScan(POTSIZE, c, a);
    printElapsedTime(StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation(), "(std::chrono Measured)");
    printArray(count, c, true);
    printCmpLenResult(count, expectedCount, b, c);

    zeroArray(POTSIZE, c);
    printDesc("work-efficient compact, power-of-two");
    count = StreamCompaction::Efficient::compact(POTSIZE, c, a);
    printElapsedTime(StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
    printArray(count, c, true);
    printCmpLenResult(count, expectedCount, b, c);

    zeroArray(POTSIZE, c);
    printDesc("work-efficient compact, non-power-of-two");
    count = StreamCompaction::Efficient::compact(NPOT, c, a);
    printElapsedTime(StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
    printArray(count, c, true);
    printCmpLenResult(count, expectedNPOT, b, c);

    zeroArray(POTSIZE, c);
    printDesc("work-efficient compact using shared memory, power-of-two");
    count = StreamCompaction::EfficientShare::compact(POTSIZE, c, a);
    printElapsedTime(StreamCompaction::EfficientShare::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
    printArray(count, c, true);
    printCmpLenResult(count, expectedCount, b, c);

    zeroArray(POTSIZE, c);
    printDesc("work-efficient compact using shared memory, non-power-of-two");
    count = StreamCompaction::EfficientShare::compact(NPOT, c, a);
    printElapsedTime(StreamCompaction::EfficientShare::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
    printArray(count, c, true);
    printCmpLenResult(count, expectedNPOT, b, c);

    

    delete[] a;
    delete[] b;
    delete[] c;
    system("pause"); // stop Win32 console from closing on exit
    
}

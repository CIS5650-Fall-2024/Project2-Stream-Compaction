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
#include "testing_helpers.hpp"

#define MAX_SIZE 1 << 18

void scanTests(const int SIZE, const int NPOT, int *a, int *b, int *c, int *d) {
    genArray(SIZE - 1, a, 50);  // Leave a 0 at the end to test that edge case
    a[SIZE - 1] = 0;
    printArray(SIZE, a, true);

    // initialize b using StreamCompaction::CPU::scan you implement
    // We use b for further comparison. Make sure your StreamCompaction::CPU::scan is correct.
    // At first all cases passed because b && c are all zeroes.
    zeroArray(SIZE, b);
    printDesc("cpu scan, power-of-two, serial");
    StreamCompaction::CPU::scan(SIZE, b, a, /*simulateGPUScan=*/false);
    printElapsedTime(StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation(), "(std::chrono Measured)");
    printArray(SIZE, b, true);

    zeroArray(SIZE, c);
    printDesc("cpu scan, non-power-of-two, serial");
    StreamCompaction::CPU::scan(NPOT, c, a, /*simulateGPUScan=*/false);
    printElapsedTime(StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation(), "(std::chrono Measured)");
    if (!printCmpResult(NPOT, b, c)) printArray(NPOT, c, true);

    zeroArray(SIZE, d);
    printDesc("cpu scan, power-of-two, simulated GPU scan");
    StreamCompaction::CPU::scan(SIZE, d, a, /*simulateGPUScan=*/true);
    printElapsedTime(StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation(), "(std::chrono Measured)");
    if (!printCmpResult(SIZE, b, d)) printArray(SIZE, d, true);

    zeroArray(SIZE, d);
    printDesc("cpu scan, non-power-of-two, simulated GPU scan");
    StreamCompaction::CPU::scan(NPOT, d, a, /*simulateGPUScan=*/true);
    printElapsedTime(StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation(), "(std::chrono Measured)");
    if (!printCmpResult(NPOT, c, d)) printArray(NPOT, d, true);

    zeroArray(SIZE, c);
    printDesc("naive scan, power-of-two");
    StreamCompaction::Naive::scan(SIZE, c, a);
    printElapsedTime(StreamCompaction::Naive::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
    if (!printCmpResult(SIZE, b, c)) printArray(SIZE, c, true);

    // For bug-finding only: Array of 1s to help find bugs in stream compaction or scan
    // onesArray(SIZE, c);
    // printDesc("1s array for finding bugs");
    // StreamCompaction::Naive::scan(SIZE, c, a);
    // printArray(SIZE, c, true);

    zeroArray(SIZE, c);
    printDesc("naive scan, non-power-of-two");
    StreamCompaction::Naive::scan(NPOT, c, a);
    printElapsedTime(StreamCompaction::Naive::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
    if (!printCmpResult(NPOT, b, c)) printArray(SIZE, c, true);

    // zeroArray(SIZE, c);
    // zeroArray(SIZE, d);
    // printDesc("parallel reduction, power-of-two");
    // StreamCompaction::Efficient::reduce(SIZE, d, a);
    // StreamCompaction::Efficient::scan(SIZE, c, a);
    // printElapsedTime(StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
    // printArray(SIZE, d, true);
    // printArray(SIZE, c, true);
    // printCmpResult(SIZE, c, d);

    // zeroArray(SIZE, c);
    // zeroArray(SIZE, d);
    // printDesc("parallel reduction, non-power-of-two");
    // StreamCompaction::Efficient::reduce(NPOT, d, a);
    // StreamCompaction::Efficient::scan(NPOT, c, a);
    // printElapsedTime(StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
    // printArray(NPOT, d, true);
    // printArray(NPOT, c, true);
    // printCmpResult(NPOT, c, d);

    zeroArray(SIZE, c);
    printDesc("work-efficient scan, power-of-two");
    StreamCompaction::Efficient::scan(SIZE, c, a);
    printElapsedTime(StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
    if (!printCmpResult(SIZE, b, c)) printArray(SIZE, c, true);

    zeroArray(SIZE, c);
    printDesc("work-efficient scan, non-power-of-two");
    StreamCompaction::Efficient::scan(NPOT, c, a);
    printElapsedTime(StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
    if (!printCmpResult(NPOT, b, c)) printArray(NPOT, c, true);

    zeroArray(SIZE, c);
    printDesc("thrust scan, power-of-two");
    StreamCompaction::Thrust::scan(SIZE, c, a);
    printElapsedTime(StreamCompaction::Thrust::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
    if (!printCmpResult(SIZE, b, c)) printArray(SIZE, c, true);

    zeroArray(SIZE, c);
    printDesc("thrust scan, non-power-of-two");
    StreamCompaction::Thrust::scan(NPOT, c, a);
    printElapsedTime(StreamCompaction::Thrust::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
    if (!printCmpResult(NPOT, b, c)) printArray(NPOT, c, true);
}

void compactionTests(const int SIZE, const int NPOT, int *a, int *b, int *c, int *d) {
    genArray(SIZE - 1, a, 4);  // Leave a 0 at the end to test that edge case
    a[SIZE - 1] = 0;
    printArray(SIZE, a, true);

    int count, expectedCount, expectedNPOT;

    // initialize b using StreamCompaction::CPU::compactWithoutScan you implement
    // We use b for further comparison. Make sure your StreamCompaction::CPU::compactWithoutScan is correct.
    zeroArray(SIZE, b);
    printDesc("cpu compact without scan, power-of-two");
    count = StreamCompaction::CPU::compactWithoutScan(SIZE, b, a);
    printElapsedTime(StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation(), "(std::chrono Measured)");
    expectedCount = count;
    if (!printCmpLenResult(count, expectedCount, b, b)) printArray(count, b, true);

    zeroArray(SIZE, c);
    printDesc("cpu compact without scan, non-power-of-two");
    count = StreamCompaction::CPU::compactWithoutScan(NPOT, c, a);
    printElapsedTime(StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation(), "(std::chrono Measured)");
    expectedNPOT = count;
    if (!printCmpLenResult(count, expectedNPOT, b, c)) printArray(count, c, true);

    zeroArray(SIZE, c);
    printDesc("cpu compact with scan");
    count = StreamCompaction::CPU::compactWithScan(SIZE, c, a);
    printElapsedTime(StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation(), "(std::chrono Measured)");
    if (!printCmpLenResult(count, expectedCount, b, c)) printArray(count, c, true);

    zeroArray(SIZE, c);
    printDesc("work-efficient compact, power-of-two");
    count = StreamCompaction::Efficient::compact(SIZE, c, a);
    printElapsedTime(StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
    if (!printCmpLenResult(count, expectedCount, b, c)) printArray(count, c, true);

    zeroArray(SIZE, c);
    printDesc("work-efficient compact, non-power-of-two");
    count = StreamCompaction::Efficient::compact(NPOT, c, a);
    printElapsedTime(StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
    if (!printCmpLenResult(count, expectedNPOT, b, c)) printArray(count, c, true);
}

int main(int argc, char* argv[]) {
    printf("\n");
    printf("****************\n");
    printf("** SCAN TESTS **\n");
    printf("****************\n");
    
    // TODO: crashes for SIZE 1 and 2.
    for (int SIZE = 4; SIZE <= MAX_SIZE; SIZE <<= 1) {
        // TODO: work-efficient scan fails for size < 256.
        // TODO: fails for 1 << 1.
        // Non-power-of-two.
        const int NPOT = SIZE - 3;

        printf("\n");
        printf("***********************************************************************\n");
        printf("**SCAN, SIZE = 2^%d = %d NPOT = %d **\n", ilog2(SIZE), SIZE, NPOT);
        printf("***********************************************************************\n");

        int *a = new int[SIZE];
        int *b = new int[SIZE];
        int *c = new int[SIZE];
        int *d = new int[SIZE];
        
        scanTests(SIZE, NPOT, a, b, c, d);

        delete[] d;
        delete[] c;
        delete[] b;
        delete[] a;
    }

    printf("\n");
    printf("*****************************\n");
    printf("** STREAM COMPACTION TESTS **\n");
    printf("*****************************\n");

    // TODO: crashes for SIZE 1 and 2.
    for (int SIZE = 4; SIZE <= MAX_SIZE; SIZE <<= 1) {
        // TODO: fails for 1 << 1.
        // Non-power-of-two.
        const int NPOT = SIZE - 3;

        printf("\n");
        printf("***********************************************************************\n");
        printf("**COMPACT, SIZE = 2^%d = %d NPOT = %d **\n", ilog2(SIZE), SIZE, NPOT);
        printf("***********************************************************************\n");

        int *a = new int[SIZE];
        int *b = new int[SIZE];
        int *c = new int[SIZE];
        int *d = new int[SIZE];
        
        compactionTests(SIZE, NPOT, a, b, c, d);

        delete[] d;
        delete[] c;
        delete[] b;
        delete[] a;
    }

     // Stop Win32 console from closing on exit.
    system("pause");
}

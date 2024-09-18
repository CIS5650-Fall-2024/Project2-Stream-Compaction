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
#include <stream_compaction/radix.h>
#include "testing_helpers.hpp"

// Log writing code
#include <iostream>
#include <fstream>

#define SAVE_LOG 0

const int SIZE = 1 << 20; // feel free to change the size of array
const int NPOT = SIZE - 3; // Non-Power-Of-Two
int *a = new int[SIZE];
int *b = new int[SIZE];
int *c = new int[SIZE];
int *d = new int[SIZE];

float elapsedTime;

int main(int argc, char* argv[]) {

    // Log writing code
    std::ofstream logFile;
    if (SAVE_LOG) {
        logFile.open("log.txt");
    }

    // Scan tests

    printf("\n");
    printf("****************\n");
    printf("** SCAN TESTS **\n");
    printf("****************\n");

    genArray(SIZE - 1, a, 50);  // Leave a 0 at the end to test that edge case
    a[SIZE - 1] = 0;
    printArray(SIZE, a, true);

    // initialize b using StreamCompaction::CPU::scan you implement
    // We use b for further comparison. Make sure your StreamCompaction::CPU::scan is correct.
    // At first all cases passed because b && c are all zeroes.
    zeroArray(SIZE, b);
    printDesc("cpu scan, power-of-two");
    StreamCompaction::CPU::scan(SIZE, b, a);
    elapsedTime = StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation();
    if (SAVE_LOG) logFile << elapsedTime << "\n";
    printElapsedTime(elapsedTime, "(std::chrono Measured)");
    printArray(SIZE, b, true);

    zeroArray(SIZE, c);
    printDesc("cpu scan, non-power-of-two");
    StreamCompaction::CPU::scan(NPOT, c, a);
    elapsedTime = StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation();
    if (SAVE_LOG) logFile << elapsedTime << "\n";
    printElapsedTime(elapsedTime, "(std::chrono Measured)");
    printArray(NPOT, b, true);
    printCmpResult(NPOT, b, c);

    zeroArray(SIZE, c);
    printDesc("naive scan, power-of-two");
    StreamCompaction::Naive::scan(SIZE, c, a);
    elapsedTime = StreamCompaction::Naive::timer().getGpuElapsedTimeForPreviousOperation();
    if (SAVE_LOG) logFile << elapsedTime << "\n";
    printElapsedTime(elapsedTime, "(CUDA Measured)");
    //printArray(SIZE, c, true);
    printCmpResult(SIZE, b, c);

    /* For bug-finding only: Array of 1s to help find bugs in stream compaction or scan
    onesArray(SIZE, c);
    printDesc("1s array for finding bugs");
    StreamCompaction::Naive::scan(SIZE, c, a);
    printArray(SIZE, c, true); */

    zeroArray(SIZE, c);
    printDesc("naive scan, non-power-of-two");
    StreamCompaction::Naive::scan(NPOT, c, a);
    elapsedTime = StreamCompaction::Naive::timer().getGpuElapsedTimeForPreviousOperation();
    if (SAVE_LOG) logFile << elapsedTime << "\n";
    printElapsedTime(elapsedTime, "(CUDA Measured)");
    //printArray(SIZE, c, true);
    printCmpResult(NPOT, b, c);

    zeroArray(SIZE, c);
    printDesc("work-efficient scan, power-of-two");
    StreamCompaction::Efficient::scan(SIZE, c, a);
    elapsedTime = StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation();
    if (SAVE_LOG) logFile << elapsedTime << "\n";
    printElapsedTime(elapsedTime, "(CUDA Measured)");
    //printArray(SIZE, c, true);
    printCmpResult(SIZE, b, c);

    zeroArray(SIZE, c);
    printDesc("work-efficient scan, non-power-of-two");
    StreamCompaction::Efficient::scan(NPOT, c, a);
    elapsedTime = StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation();
    if (SAVE_LOG) logFile << elapsedTime << "\n";
    printElapsedTime(elapsedTime, "(CUDA Measured)");
    //printArray(NPOT, c, true);
    printCmpResult(NPOT, b, c);

    zeroArray(SIZE, c);
    printDesc("thrust scan, power-of-two");
    StreamCompaction::Thrust::scan(SIZE, c, a);
    elapsedTime = StreamCompaction::Thrust::timer().getGpuElapsedTimeForPreviousOperation();
    if (SAVE_LOG) logFile << elapsedTime << "\n";
    printElapsedTime(elapsedTime, "(CUDA Measured)");
    //printArray(SIZE, c, true);
    printCmpResult(SIZE, b, c);

    zeroArray(SIZE, c);
    printDesc("thrust scan, non-power-of-two");
    StreamCompaction::Thrust::scan(NPOT, c, a);
    elapsedTime = StreamCompaction::Thrust::timer().getGpuElapsedTimeForPreviousOperation();
    if (SAVE_LOG) logFile << elapsedTime << "\n";
    printElapsedTime(elapsedTime, "(CUDA Measured)");
    //printArray(NPOT, c, true);
    printCmpResult(NPOT, b, c);

    printf("\n");
    printf("*****************************\n");
    printf("** STREAM COMPACTION TESTS **\n");
    printf("*****************************\n");

    // Compaction tests

    genArray(SIZE - 1, a, 4);  // Leave a 0 at the end to test that edge case
    a[SIZE - 1] = 0;
    printArray(SIZE, a, true);

    int count, expectedCount, expectedNPOT;

    // initialize b using StreamCompaction::CPU::compactWithoutScan you implement
    // We use b for further comparison. Make sure your StreamCompaction::CPU::compactWithoutScan is correct.
    zeroArray(SIZE, b);
    printDesc("cpu compact without scan, power-of-two");
    count = StreamCompaction::CPU::compactWithoutScan(SIZE, b, a);
    elapsedTime = StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation();
    if (SAVE_LOG) logFile << elapsedTime << "\n";
    printElapsedTime(elapsedTime, "(std::chrono Measured)");
    expectedCount = count;
    printArray(count, b, true);
    printCmpLenResult(count, expectedCount, b, b);

    zeroArray(SIZE, c);
    printDesc("cpu compact without scan, non-power-of-two");
    count = StreamCompaction::CPU::compactWithoutScan(NPOT, c, a);
    elapsedTime = StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation();
    if (SAVE_LOG) logFile << elapsedTime << "\n";
    printElapsedTime(elapsedTime, "(std::chrono Measured)");
    expectedNPOT = count;
    printArray(count, c, true);
    printCmpLenResult(count, expectedNPOT, b, c);

    zeroArray(SIZE, c);
    printDesc("cpu compact with scan");
    count = StreamCompaction::CPU::compactWithScan(SIZE, c, a);
    elapsedTime = StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation();
    if (SAVE_LOG) logFile << elapsedTime << "\n";
    printElapsedTime(elapsedTime, "(std::chrono Measured)");
    printArray(count, c, true);
    printCmpLenResult(count, expectedCount, b, c);

    zeroArray(SIZE, c);
    printDesc("work-efficient compact, power-of-two");
    count = StreamCompaction::Efficient::compact(SIZE, c, a);
    elapsedTime = StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation();
    if (SAVE_LOG) logFile << elapsedTime << "\n";
    printElapsedTime(elapsedTime, "(CUDA Measured)");
    //printArray(count, c, true);
    printCmpLenResult(count, expectedCount, b, c);

    zeroArray(SIZE, c);
    printDesc("work-efficient compact, non-power-of-two");
    count = StreamCompaction::Efficient::compact(NPOT, c, a);
    elapsedTime = StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation();
    if (SAVE_LOG) logFile << elapsedTime << "\n";
    printElapsedTime(elapsedTime, "(CUDA Measured)");
    //printArray(count, c, true);
    printCmpLenResult(count, expectedNPOT, b, c);

    printf("\n");
    printf("*****************************\n");
    printf("** RADIX SORT TESTS **\n");
    printf("*****************************\n");

    genArray(SIZE, a, SIZE);
    printArray(SIZE, a, true);

    zeroArray(SIZE, b);
    printDesc("cpu sort, power-of-two");
    StreamCompaction::CPU::sort(SIZE, b, a);
    elapsedTime = StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation();
    if (SAVE_LOG) logFile << elapsedTime << "\n";
    printElapsedTime(elapsedTime, "(std::chrono Measured)");
    printArray(SIZE, b, true);

    zeroArray(SIZE, c);
    printDesc("cpu sort, non-power-of-two");
    StreamCompaction::CPU::sort(NPOT, c, a);
    elapsedTime = StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation();
    if (SAVE_LOG) logFile << elapsedTime << "\n";
    printElapsedTime(elapsedTime, "(std::chrono Measured)");
    printArray(NPOT, c, true);

    zeroArray(SIZE, d);
    printDesc("radix sort, power-of-two");
    StreamCompaction::Radix::radixSort(SIZE, d, a);
    elapsedTime = StreamCompaction::Radix::timer().getGpuElapsedTimeForPreviousOperation();
    if (SAVE_LOG) logFile << elapsedTime << "\n";
    printElapsedTime(elapsedTime, "(CUDA Measured)");
    printArray(SIZE, d, true);
    printCmpResult(SIZE, b, d);

    zeroArray(SIZE, d);
    printDesc("radix sort, non-power-of-two");
    StreamCompaction::Radix::radixSort(NPOT, d, a);
    elapsedTime = StreamCompaction::Radix::timer().getGpuElapsedTimeForPreviousOperation();
    if (SAVE_LOG) logFile << elapsedTime << "\n";
    printElapsedTime(elapsedTime, "(CUDA Measured)");
    printArray(SIZE, d, true);
    printCmpResult(NPOT, c, d);

    if (SAVE_LOG)
        logFile.close();

    system("pause"); // stop Win32 console from closing on exit
    delete[] a;
    delete[] b;
    delete[] c;
}

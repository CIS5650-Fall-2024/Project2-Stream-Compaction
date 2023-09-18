/**
 * @file      main.cpp
 * @brief     Stream compaction test program
 * @authors   Kai Ninomiya
 * @date      2015
 * @copyright University of Pennsylvania
 */

#include <cstdio>
#include <vector>
#include <numeric>
#include <stream_compaction/cpu.h>
#include <stream_compaction/naive.h>
#include <stream_compaction/efficient.h>
#include <stream_compaction/thrust.h>
#include "testing_helpers.hpp"

const int SIZE = 1 << 3; // feel free to change the size of array
const int NPOT = SIZE - 3; // Non-Power-Of-Two
int* a = new int[SIZE];
int* b = new int[SIZE];
int* c = new int[SIZE];

#define TEST 1 // 1: running test; 0: perform anaylsis

#if TEST
int main(int argc, char* argv[]) {
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
    printElapsedTime(StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation(), "(std::chrono Measured)");
    printArray(SIZE, b, true);

    zeroArray(SIZE, c);
    printDesc("cpu scan, non-power-of-two");
    StreamCompaction::CPU::scan(NPOT, c, a);
    printElapsedTime(StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation(), "(std::chrono Measured)");
    printArray(NPOT, b, true);
    printCmpResult(NPOT, b, c);

    zeroArray(SIZE, c);
    printDesc("naive scan, power-of-two");
    StreamCompaction::Naive::scan(SIZE, c, a);
    printElapsedTime(StreamCompaction::Naive::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
    printArray(SIZE, c, true);
    printCmpResult(SIZE, b, c);

    // For bug-finding only: Array of 1s to help find bugs in stream compaction or scan
    /*onesArray(SIZE, c);
    printDesc("1s array for finding bugs");
    StreamCompaction::Naive::scan(SIZE, c, a);
    printArray(SIZE, c, true);*/

    zeroArray(SIZE, c);
    printDesc("naive scan, non-power-of-two");
    StreamCompaction::Naive::scan(NPOT, c, a);
    printElapsedTime(StreamCompaction::Naive::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
    printArray(SIZE, c, true);
    printCmpResult(NPOT, b, c);

    zeroArray(SIZE, c);
    printDesc("work-efficient scan, power-of-two");
    StreamCompaction::Efficient::scan(SIZE, c, a);
    printElapsedTime(StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
    printArray(SIZE, c, true);
    printCmpResult(SIZE, b, c);

    zeroArray(SIZE, c);
    printDesc("work-efficient scan, non-power-of-two");
    StreamCompaction::Efficient::scan(NPOT, c, a);
    printElapsedTime(StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
    printArray(NPOT, c, true);
    printCmpResult(NPOT, b, c);

    zeroArray(SIZE, c);
    printDesc("thrust scan, power-of-two");
    StreamCompaction::Thrust::scan(SIZE, c, a);
    printElapsedTime(StreamCompaction::Thrust::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
    //printArray(SIZE, c, true);
    printCmpResult(SIZE, b, c);

    zeroArray(SIZE, c);
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
    printArray(count, b, true);
    printCmpLenResult(count, expectedCount, b, b);

    zeroArray(SIZE, c);
    printDesc("cpu compact without scan, non-power-of-two");
    count = StreamCompaction::CPU::compactWithoutScan(NPOT, c, a);
    printElapsedTime(StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation(), "(std::chrono Measured)");
    expectedNPOT = count;
    printArray(count, c, true);
    printCmpLenResult(count, expectedNPOT, b, c);

    zeroArray(SIZE, c);
    printDesc("cpu compact with scan");
    count = StreamCompaction::CPU::compactWithScan(SIZE, c, a);
    printElapsedTime(StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation(), "(std::chrono Measured)");
    printArray(count, c, true);
    printCmpLenResult(count, expectedCount, b, c);

    zeroArray(SIZE, c);
    printDesc("work-efficient compact, power-of-two");
    count = StreamCompaction::Efficient::compact(SIZE, c, a);
    printElapsedTime(StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
    printArray(count, c, true);
    printCmpLenResult(count, expectedCount, b, c);

    zeroArray(SIZE, c);
    printDesc("work-efficient compact, non-power-of-two");
    count = StreamCompaction::Efficient::compact(NPOT, c, a);
    printElapsedTime(StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
    printArray(count, c, true);
    printCmpLenResult(count, expectedNPOT, b, c);

    system("pause"); // stop Win32 console from closing on exit
    delete[] a;
    delete[] b;
    delete[] c;
}

#else
const int NUM_TESTS = 10;

double computeAverage(double* arr, int size) {
    double sum = 0.0;
    for (int i = 0; i < size; ++i) {
        sum += arr[i];
    }
    return sum / size;
}

void testScan() {
    // Scan tests

    printf("\n");
    printf("****************\n");
    printf("** SCAN TESTS **\n");
    printf("****************\n");

    double* cpuScanTimes = new double[NUM_TESTS];
    double* naiveScanTimes = new double[NUM_TESTS];
    double* workEfficientScanTimes = new double[NUM_TESTS];
    double* thrustScanTimes = new double[NUM_TESTS];

    for (int i = 0; i < NUM_TESTS; ++i) {
        genArray(SIZE - 1, a, 50);  // Leave a 0 at the end to test that edge case
        a[SIZE - 1] = 0;

        zeroArray(SIZE, b);
        printDesc("cpu scan, power-of-two");
        StreamCompaction::CPU::scan(SIZE, b, a);
        cpuScanTimes[i] = StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation();
        
        zeroArray(SIZE, c);
        printDesc("cpu scan, not power-of-two");
        StreamCompaction::CPU::scan(NPOT, c, a);
        evalCmpResult(NPOT, b, c);

        zeroArray(SIZE, c);
        StreamCompaction::Naive::scan(SIZE, c, a);
        naiveScanTimes[i] = StreamCompaction::Naive::timer().getGpuElapsedTimeForPreviousOperation();
        evalCmpResult(SIZE, b, c);

        zeroArray(SIZE, c);
        printDesc("naive scan, not power-of-two");
        StreamCompaction::Naive::scan(NPOT, c, a);
        evalCmpResult(NPOT, b, c);

        zeroArray(SIZE, c);
        printDesc("work-efficient scan, power-of-two");
        StreamCompaction::Efficient::scan(SIZE, c, a);
        workEfficientScanTimes[i] = StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation();
        evalCmpResult(SIZE, b, c);

        zeroArray(SIZE, c);
        printDesc("work-efficient scan, not power-of-two");
        StreamCompaction::Efficient::scan(NPOT, c, a);
        evalCmpResult(NPOT, b, c);

        zeroArray(SIZE, c);
        printDesc("thrust scan, power-of-two");
        StreamCompaction::Thrust::scan(SIZE, c, a);
        thrustScanTimes[i] = StreamCompaction::Thrust::timer().getGpuElapsedTimeForPreviousOperation();
        evalCmpResult(SIZE, b, c);

        zeroArray(SIZE, c);
        printDesc("thrust scan, not power-of-two");
        StreamCompaction::Thrust::scan(NPOT, c, a);
        evalCmpResult(NPOT, b, c);
    }

    printDesc("cpu scan, power-of-two");
    printDoubleArray(NUM_TESTS, cpuScanTimes, true);
    printf("%5f \n", computeAverage(cpuScanTimes, NUM_TESTS));

    printDesc("naive scan,power-of-two");
    printDoubleArray(NUM_TESTS, naiveScanTimes, true);
    printf("%5f \n", computeAverage(naiveScanTimes, NUM_TESTS));

    printDesc("work-efficient scan, power-of-two");
    printDoubleArray(NUM_TESTS, workEfficientScanTimes, true);
    printf("%5f \n", computeAverage(workEfficientScanTimes, NUM_TESTS));

    printDesc("thrust scan, power-of-two");
    printDoubleArray(NUM_TESTS, thrustScanTimes, true);
    printf("%5f \n", computeAverage(thrustScanTimes, NUM_TESTS));

   
    delete[] cpuScanTimes;
    delete[] naiveScanTimes;
    delete[] workEfficientScanTimes;
    delete[] thrustScanTimes;
}

void testCompact() {
    printf("\n");
    printf("*****************************\n");
    printf("** STREAM COMPACTION TESTS **\n");
    printf("*****************************\n");

    // Compaction tests

    double* cpuCompact = new double[NUM_TESTS];
    double* cpuCompactWithScan = new double[NUM_TESTS];
    double* workEfficientCompact = new double[NUM_TESTS];

    for (int i = 0; i < NUM_TESTS; ++i) {
        genArray(SIZE - 1, a, 4);  // Leave a 0 at the end to test that edge case
        a[SIZE - 1] = 0;
        //printArray(SIZE, a, true);

        int count, expectedCount, expectedNPOT;

        // initialize b using StreamCompaction::CPU::compactWithoutScan you implement
        // We use b for further comparison. Make sure your StreamCompaction::CPU::compactWithoutScan is correct.
        zeroArray(SIZE, b);
        printDesc("cpu compact without scan, power-of-two");
        count = StreamCompaction::CPU::compactWithoutScan(SIZE, b, a);
        cpuCompact[i] = StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation();
        expectedCount = count;
        evalCmpLenResult(count, expectedCount, b, b);

        zeroArray(SIZE, c);
        printDesc("cpu compact without scan, non-power-of-two");
        count = StreamCompaction::CPU::compactWithoutScan(NPOT, c, a);
        expectedNPOT = count;
        evalCmpLenResult(count, expectedNPOT, b, c);

        zeroArray(SIZE, c);
        printDesc("cpu compact with scan");
        count = StreamCompaction::CPU::compactWithScan(SIZE, c, a);
        cpuCompactWithScan[i] = StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation();
        evalCmpLenResult(count, expectedCount, b, c);

        zeroArray(SIZE, c);
        printDesc("work-efficient compact, power-of-two");
        count = StreamCompaction::Efficient::compact(SIZE, c, a);
        workEfficientCompact[i] = StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation();
        evalCmpLenResult(count, expectedCount, b, c);

        zeroArray(SIZE, c);
        printDesc("work-efficient compact, non-power-of-two");
        count = StreamCompaction::Efficient::compact(NPOT, c, a);
        evalCmpLenResult(count, expectedNPOT, b, c);
    }

    printDesc("cpu compact without scan, power-of-two");
    printDoubleArray(NUM_TESTS, cpuCompact, true);
    printf("%5f \n", computeAverage(cpuCompact, NUM_TESTS));

    printDesc("cpu compact with scan, power-of-two");
    printDoubleArray(NUM_TESTS, cpuCompactWithScan, true);
    printf("%5f \n", computeAverage(cpuCompactWithScan, NUM_TESTS));

    printDesc("work-efficient compact, power-of-two");
    printDoubleArray(NUM_TESTS, workEfficientCompact, true);
    printf("%5f \n", computeAverage(workEfficientCompact, NUM_TESTS));

    delete[] cpuCompact;
    delete[] cpuCompactWithScan;
    delete[] workEfficientCompact;
}

int main(int argc, char* argv[]) {
    testScan();
    testCompact();

    system("pause"); // stop Win32 console from closing on exit
    delete[] a;
    delete[] b;
    delete[] c;
}
#endif
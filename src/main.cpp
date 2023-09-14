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

#define TEST_SCAN 1
#define TEST_STREAM_COMPACTION 1

#define TEST_NPOT 1

#define PRINT_RESULTS 1

#define GPU_TIMING_ONLY 0

const int SIZE = 1 << 25;
const int NPOT = SIZE - 3; // Non-Power-Of-Two
int *a = new int[SIZE];
int *b = new int[SIZE];
int *c = new int[SIZE];

int main(int argc, char* argv[]) {
#if TEST_SCAN
    printf("\n");
    printf("****************\n");
    printf("** SCAN TESTS **\n");
    printf("****************\n");

    genArray(SIZE - 1, a, 50);  // Leave a 0 at the end to test that edge case
    a[SIZE - 1] = 0;

    printArray(SIZE, a, true);

#if !GPU_TIMING_ONLY
    // initialize b using StreamCompaction::CPU::scan you implement
    // We use b for further comparison. Make sure your StreamCompaction::CPU::scan is correct.
    // At first all cases passed because b && c are all zeroes.
    zeroArray(SIZE, b);
    printDesc("cpu scan, power-of-two");
    printElapsedTime(testMultipleTimes<float>(
        [&]() { StreamCompaction::CPU::scan(SIZE, b, a); },
        []() { return StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation(); }
    ), "(std::chrono Measured)");
#if PRINT_RESULTS
    printArray(SIZE, b, true);
#endif

#if TEST_NPOT
    zeroArray(SIZE, c);
    printDesc("cpu scan, non-power-of-two");
    printElapsedTime(testMultipleTimes<float>(
        [&]() { StreamCompaction::CPU::scan(NPOT, c, a); },
        []() { return StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation(); }
    ), "(std::chrono Measured)");
#if PRINT_RESULTS
    printArray(NPOT, b, true);
    printCmpResult(NPOT, b, c);
#endif
#endif
#endif

    zeroArray(SIZE, c);
    printDesc("naive scan, power-of-two");
    printElapsedTime(testMultipleTimes<float>(
        [&]() { StreamCompaction::Naive::scan(SIZE, c, a); },
        []() { return StreamCompaction::Naive::timer().getGpuElapsedTimeForPreviousOperation(); }
    ), "(CUDA Measured)");
#if PRINT_RESULTS
    //printArray(SIZE, c, true);
    printCmpResult(SIZE, b, c);
#endif

    /* For bug-finding only: Array of 1s to help find bugs in stream compaction or scan
    onesArray(SIZE, c);
    printDesc("1s array for finding bugs");
    StreamCompaction::Naive::scan(SIZE, c, a);
    printArray(SIZE, c, true); */

#if !GPU_TIMING_ONLY
#if TEST_NPOT
    zeroArray(SIZE, c);
    printDesc("naive scan, non-power-of-two");
    printElapsedTime(testMultipleTimes<float>(
        [&]() { StreamCompaction::Naive::scan(NPOT, c, a); },
        []() { return StreamCompaction::Naive::timer().getGpuElapsedTimeForPreviousOperation(); }
    ), "(CUDA Measured)");
#if PRINT_RESULTS
    //printArray(SIZE, c, true);
    printCmpResult(NPOT, b, c);
#endif
#endif
#endif

    zeroArray(SIZE, c);
    printDesc("work-efficient scan, power-of-two");
    printElapsedTime(testMultipleTimes<float>(
        [&]() { StreamCompaction::Efficient::scan(SIZE, c, a); },
        []() { return StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation(); }
    ), "(CUDA Measured)");
    //printArray(SIZE, c, true);
    printCmpResult(SIZE, b, c);

#if !GPU_TIMING_ONLY
#if TEST_NPOT
    zeroArray(SIZE, c);
    printDesc("work-efficient scan, non-power-of-two");
    printElapsedTime(testMultipleTimes<float>(
        [&]() { StreamCompaction::Efficient::scan(NPOT, c, a); },
        []() { return StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation(); }
    ), "(CUDA Measured)");
#if PRINT_RESULTS
    //printArray(NPOT, c, true);
    printCmpResult(NPOT, b, c);
#endif
#endif
#endif

    zeroArray(SIZE, c);
    printDesc("thrust scan, power-of-two");
    printElapsedTime(testMultipleTimes<float>(
        [&]() { StreamCompaction::Thrust::scan(SIZE, c, a); },
        []() { return StreamCompaction::Thrust::timer().getGpuElapsedTimeForPreviousOperation(); }
    ), "(CUDA Measured)");
#if PRINT_RESULTS
    //printArray(SIZE, c, true);
    printCmpResult(SIZE, b, c);
#endif

#if !GPU_TIMING_ONLY
#if TEST_NPOT
    zeroArray(SIZE, c);
    printDesc("thrust scan, non-power-of-two");
    printElapsedTime(testMultipleTimes<float>(
        [&]() { StreamCompaction::Thrust::scan(NPOT, c, a); },
        []() { return StreamCompaction::Thrust::timer().getGpuElapsedTimeForPreviousOperation(); }
    ), "(CUDA Measured)");
#if PRINT_RESULTS
    //printArray(NPOT, c, true);
    printCmpResult(NPOT, b, c);
#endif
#endif
#endif
#endif

#if TEST_STREAM_COMPACTION
    printf("\n");
    printf("*****************************\n");
    printf("** STREAM COMPACTION TESTS **\n");
    printf("*****************************\n");

    genArray(SIZE - 1, a, 4);  // Leave a 0 at the end to test that edge case
    a[SIZE - 1] = 0;
    printArray(SIZE, a, true);

    int count, expectedCount, expectedNPOT;

#if !GPU_TIMING_ONLY
    // initialize b using StreamCompaction::CPU::compactWithoutScan you implement
    // We use b for further comparison. Make sure your StreamCompaction::CPU::compactWithoutScan is correct.
    zeroArray(SIZE, b);
    printDesc("cpu compact without scan, power-of-two");
    printElapsedTime(testMultipleTimes<float>(
        [&]() { count = StreamCompaction::CPU::compactWithoutScan(SIZE, b, a); },
        []() { return StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation(); }
    ), "(std::chrono Measured)");
    expectedCount = count;
#if PRINT_RESULTS
    printArray(count, b, true);
    printCmpLenResult(count, expectedCount, b, b);
#endif

#if TEST_NPOT
    zeroArray(SIZE, c);
    printDesc("cpu compact without scan, non-power-of-two");
    printElapsedTime(testMultipleTimes<float>(
        [&]() { count = StreamCompaction::CPU::compactWithoutScan(NPOT, c, a); },
        []() { return StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation(); }
    ), "(std::chrono Measured)");
    expectedNPOT = count;
#if PRINT_RESULTS
    printArray(count, c, true);
    printCmpLenResult(count, expectedNPOT, b, c);
#endif
#endif

    zeroArray(SIZE, c);
    printDesc("cpu compact with scan");
    printElapsedTime(testMultipleTimes<float>(
        [&]() { count = StreamCompaction::CPU::compactWithScan(SIZE, c, a); },
        []() { return StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation(); }
    ), "(std::chrono Measured)");
#if PRINT_RESULTS
    printArray(count, c, true);
    printCmpLenResult(count, expectedCount, b, c);
#endif
#endif

    zeroArray(SIZE, c);
    printDesc("work-efficient compact, power-of-two");
    printElapsedTime(testMultipleTimes<float>(
        [&]() { count = StreamCompaction::Efficient::compact(SIZE, c, a); },
        []() { return StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation(); }
    ), "(CUDA Measured)");
#if PRINT_RESULTS
    //printArray(count, c, true);
    printCmpLenResult(count, expectedCount, b, c);
#endif

#if !GPU_TIMING_ONLY
#if TEST_NPOT
    zeroArray(SIZE, c);
    printDesc("work-efficient compact, non-power-of-two");
    printElapsedTime(testMultipleTimes<float>(
        [&]() { count = StreamCompaction::Efficient::compact(NPOT, c, a); },
        []() { return StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation(); }
    ), "(CUDA Measured)");
#if PRINT_RESULTS
    //printArray(count, c, true);
    printCmpLenResult(count, expectedNPOT, b, c);
#endif
#endif
#endif
#endif

    //system("pause"); // stop Win32 console from closing on exit
    delete[] a;
    delete[] b;
    delete[] c;
}

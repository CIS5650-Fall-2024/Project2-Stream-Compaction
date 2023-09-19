/**
 * @file      main.cpp
 * @brief     Stream compaction test program
 * @authors   Kai Ninomiya
 * @date      2015
 * @copyright University of Pennsylvania
 */

#include <cstdio>
#include <stream_compaction/common.h>
#include <stream_compaction/cpu.h>
#include <stream_compaction/naive.h>
#include <stream_compaction/efficient.h>
#include <stream_compaction/radix.h>

#include <stream_compaction/thrust.h>
#include "testing_helpers.hpp"

const int SIZE = 1 << 24; // feel free to change the size of array
const int NPOT = SIZE - 3; // Non-Power-Of-Two

int *a = new int[SIZE];
int *b = new int[SIZE];
int *c = new int[SIZE];

#define CPU_TEST 1

#define SCAN 1
#define COMPACT 1
#define SORT 1
int main(int argc, char* argv[]) 
{
    StreamCompaction::Common::GetGPUInfo(true);

    int count = 1;
    
    genArray(SIZE - 1, a, 50);  // Leave a 0 at the end to test that edge case
    a[SIZE - 1] = 0;
    printArray(SIZE, a, true);

    for (int k = 1024; k <= 1024; k <<= 1)
    {
        BlockSize = k;
        for (int i = 0; i < count; ++i)
        {
            // Scan tests
#if SCAN
#if DEBUG
            printf("\n");
            printf("****************\n");
            printf("** SCAN TESTS **\n");
            printf("****************\n");
#endif
            // initialize b using thrust functions

            zeroArray(SIZE, c);
            printDesc("thrust scan, power-of-two");
            StreamCompaction::Thrust::scan(SIZE, b, a);
            printElapsedTime(StreamCompaction::Thrust::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
            printArray(SIZE, b, true);

            zeroArray(SIZE, c);
            printDesc("thrust scan, non-power-of-two");
            StreamCompaction::Thrust::scan(NPOT, c, a);
            printElapsedTime(StreamCompaction::Thrust::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
            printArray(NPOT, c, true);
            printCmpResult(NPOT, b, c);

#if CPU_TEST
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
#endif
            zeroArray(SIZE, c);
            printDesc("naive scan, power-of-two");
            StreamCompaction::Naive::scan(SIZE, c, a);
            printElapsedTime(StreamCompaction::Naive::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
            printArray(SIZE, c, true);
            printCmpResult(SIZE, b, c);

            /* For bug-finding only: Array of 1s to help find bugs in stream compaction or scan
            onesArray(SIZE, c);
            printDesc("1s array for finding bugs");
            StreamCompaction::Naive::scan(SIZE, c, a);
            printArray(SIZE, c, true); */

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
#endif

#if COMPACT
#if DEBUG
            printf("\n");
            printf("*****************************\n");
            printf("** STREAM COMPACTION TESTS **\n");
            printf("*****************************\n");
#endif
            // Compaction tests

            //genArray(SIZE - 1, a, 4);  // Leave a 0 at the end to test that edge case
            //a[SIZE - 1] = 0;
            //printArray(SIZE, a, true);

            int count, expectedCount, expectedNPOT;

            // initialize b using Thrust
            
#if CPU_TEST
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
#endif
            zeroArray(SIZE, c);
            printDesc("cpu compact with scan");
            count = StreamCompaction::CPU::compactWithScan(NPOT, c, a);
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
#endif
            //system("pause"); // stop Win32 console from closing on exit
#if SORT
#if DEBUG
            printf("\n");
            printf("*****************************\n");
            printf("** SORT TESTS **\n");
            printf("*****************************\n");
#endif     
#if CPU_TEST
            //CPU sort
            zeroArray(SIZE, c);
            printDesc("CPU sort, power-of-two");
            StreamCompaction::CPU::sort(SIZE, c, a);
            printElapsedTime(StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation(), "(std::chrono Measured)");

            zeroArray(SIZE, c);
            printDesc("CPU sort, non-power-of-two");
            StreamCompaction::CPU::sort(NPOT, c, a);
            printElapsedTime(StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation(), "(std::chrono Measured)");
#endif
            // initialize b using Thrust
            // thrust sort
            zeroArray(SIZE, b);
            printDesc("Thrust sort, power-of-two");
            StreamCompaction::Thrust::sort(SIZE, b, a);
            printElapsedTime(StreamCompaction::Thrust::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");

            // Radix sort
            zeroArray(SIZE, c);
            printDesc("Radix sort, power-of-two");
            StreamCompaction::Radix::sort(SIZE, c, a);
            printElapsedTime(StreamCompaction::Radix::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
            printArray(SIZE, c, true);
            printCmpResult(SIZE, b, c);

            zeroArray(SIZE, b);
            printDesc("Thrust sort, non-power-of-two");
            StreamCompaction::Thrust::sort(NPOT, b, a);
            printElapsedTime(StreamCompaction::Thrust::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");

            zeroArray(SIZE, c);
            printDesc("Radix sort, non-power-of-two");
            StreamCompaction::Radix::sort(NPOT, c, a);
            printElapsedTime(StreamCompaction::Radix::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
            printArray(NPOT, c, true);
            printCmpResult(NPOT, b, c);
#endif
        }
    }
    
    delete[] a;
    delete[] b;
    delete[] c;
}

/**
 * @file      main.cpp
 * @brief     Stream compaction test program
 * @authors   Kai Ninomiya
 * @date      2015
 * @copyright University of Pennsylvania
 */

#include <cstdio>
#include <iomanip>
#include <stream_compaction/cpu.h>
#include <stream_compaction/naive.h>
#include <stream_compaction/efficient.h>
#include <stream_compaction/thrust.h>
#include <stream_compaction/radix.h>
#include "testing_helpers.hpp"

#define BENCHMARK 0
#if BENCHMARK
void benchmark_scan_powerOfTwo(int repeat)
{
    std::cout << "***************************************\n";
    std::cout << "***** scan power-of-2 (repeat=" << std::setw(2) << repeat << ") *****\n";
    std::cout << "***************************************\n";
    std::cout << "Elapsed Time (ms) [Lower is Better]\n";
    std::cout << std::setw(2) << " p\t";
    std::cout << std::setw(12) << "CPU\t";
    std::cout << std::setw(12) << "GPU Naive\t";
    std::cout << std::setw(12) << "GPU Work-Eff\t";
    std::cout << std::setw(12) << "GPU Thrust\n";
    // Test on different sizes
    for (int p = 8; p < 29; ++p)
    {
        const int SIZE = 1 << p;
        int* a = new int[SIZE];
        int* b = new int[SIZE];

        float elapsedTimes[4] = {0.f, 0.f, 0.f, 0.f};

        genArray(SIZE - 1, a, 50);  // Leave a 0 at the end to test that edge case
        a[SIZE - 1] = 0;

        for (int i = 0; i < repeat; ++i)
        {
            zeroArray(SIZE, b);
            StreamCompaction::CPU::scan(SIZE, b, a);
            elapsedTimes[0] += StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation();

            zeroArray(SIZE, b);
            StreamCompaction::Naive::scan(SIZE, b, a);
            elapsedTimes[1] += StreamCompaction::Naive::timer().getGpuElapsedTimeForPreviousOperation();

            zeroArray(SIZE, b);
            StreamCompaction::Efficient::scan(SIZE, b, a);
            elapsedTimes[2] += StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation();

            zeroArray(SIZE, b);
            StreamCompaction::Thrust::scan(SIZE, b, a);
            elapsedTimes[3] += StreamCompaction::Thrust::timer().getGpuElapsedTimeForPreviousOperation();
        }

        std::cout << std::setw(2) << p << "\t";
        std::cout << std::setw(11) << elapsedTimes[0] / repeat << "\t";
        std::cout << std::setw(11) << elapsedTimes[1] / repeat << "\t";
        std::cout << std::setw(11) << elapsedTimes[2] / repeat << "\t";
        std::cout << std::setw(11) << elapsedTimes[3] / repeat << "\n";

        delete[] a;
        delete[] b;
    }
}

void benchmark_scan_nonPowerOfTwo(int repeat)
{
    std::cout << "*******************************\n";
    std::cout << "***** scan non-power-of-2 *****\n";
    std::cout << "*******************************\n";
    std::cout << "Elapsed Time (ms) [Lower is Better]\n";
    std::cout << std::setw(2) << " p\t";
    std::cout << std::setw(12) << "CPU\t";
    std::cout << std::setw(12) << "GPU Naive\t";
    std::cout << std::setw(12) << "GPU Work-Eff\t";
    std::cout << std::setw(12) << "GPU Thrust\n";
    // Test on different sizes
    for (int p = 8; p < 29; ++p)
    {
        const int SIZE = 1 << p;
        const int NPOT = SIZE - 3;
        int* a = new int[NPOT];
        int* b = new int[NPOT];

        float elapsedTimes[4] = { 0.f, 0.f, 0.f, 0.f };

        genArray(NPOT - 1, a, 50);  // Leave a 0 at the end to test that edge case
        a[NPOT - 1] = 0;

        for (int i = 0; i < repeat; ++i)
        {
            zeroArray(NPOT, b);
            StreamCompaction::CPU::scan(NPOT, b, a);
            elapsedTimes[0] += StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation();

            zeroArray(NPOT, b);
            StreamCompaction::Naive::scan(NPOT, b, a);
            elapsedTimes[1] += StreamCompaction::Naive::timer().getGpuElapsedTimeForPreviousOperation();

            zeroArray(NPOT, b);
            StreamCompaction::Efficient::scan(NPOT, b, a);
            elapsedTimes[2] += StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation();

            zeroArray(NPOT, b);
            StreamCompaction::Thrust::scan(NPOT, b, a);
            elapsedTimes[3] += StreamCompaction::Thrust::timer().getGpuElapsedTimeForPreviousOperation();
        }

        std::cout << std::setw(2) << p << "\t";
        std::cout << std::setw(11) << elapsedTimes[0] / repeat << "\t";
        std::cout << std::setw(11) << elapsedTimes[1] / repeat << "\t";
        std::cout << std::setw(11) << elapsedTimes[2] / repeat << "\t";
        std::cout << std::setw(11) << elapsedTimes[3] / repeat << "\n";

        delete[] a;
        delete[] b;
    }
}

void benchmark_stream_compaction(int repeat)
{
    std::cout << "********************************************\n";
    std::cout << "***** compaction power-of-2 (repeat=" << std::setw(2) << repeat << ") *****\n";
    std::cout << "********************************************\n";
    std::cout << "Elapsed Time (ms) [Lower is Better]\n";
    std::cout << std::setw(2) << " p\t";
    std::cout << std::setw(12) << "CPU Compaction w/o. Scan\t";
    std::cout << std::setw(12) << "CPU Compaction w/. Scan\t";
    std::cout << std::setw(12) << "GPU Compaction with Work-Eff Scan\t";
    std::cout << std::setw(12) << "GPU Compaction with thrust::remove_if\n";
    // Test on different sizes
    for (int p = 8; p < 26; ++p)
    {
        const int SIZE = 1 << p;
        int* a = new int[SIZE];
        int* b = new int[SIZE];

        float elapsedTimes[4] = { 0.f, 0.f, 0.f, 0.f};

        genArray(SIZE - 1, a, 50);  // Leave a 0 at the end to test that edge case
        a[SIZE - 1] = 0;

        for (int i = 0; i < repeat; ++i)
        {
            zeroArray(SIZE, b);
            StreamCompaction::CPU::compactWithoutScan(SIZE, b, a);
            elapsedTimes[0] += StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation();

            zeroArray(SIZE, b);
            StreamCompaction::CPU::compactWithScan(SIZE, b, a);
            elapsedTimes[1] += StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation();

            zeroArray(SIZE, b);
            StreamCompaction::Efficient::compact(SIZE, b, a);
            elapsedTimes[2] += StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation();

            zeroArray(SIZE, b);
            StreamCompaction::Thrust::compact(SIZE, b, a);
            elapsedTimes[3] += StreamCompaction::Thrust::timer().getGpuElapsedTimeForPreviousOperation();
        }

        std::cout << std::setw(2) << p << "\t";
        std::cout << std::setw(11) << elapsedTimes[0] / repeat << "\t";
        std::cout << std::setw(11) << elapsedTimes[1] / repeat << "\t";
        std::cout << std::setw(11) << elapsedTimes[2] / repeat << "\t";
        std::cout << std::setw(11) << elapsedTimes[3] / repeat << "\n";

        delete[] a;
        delete[] b;
    }
}

void benchmark_radix_sort(int repeat)
{
    std::cout << "*************************************\n";
    std::cout << "******* radix sort benchmarks *******\n";
    std::cout << "*************************************\n";
    std::cout << "Elapsed Time (ms) [Lower is Better]\n";
    std::cout << std::setw(2) << " p\t";
    std::cout << std::setw(20) << "std::stable_sort\t";
    std::cout << std::setw(20) << "Radix::sort 31-bits\t";
    std::cout << std::setw(20) << "Radix::sort 15-bits\t";
    std::cout << std::setw(20) << "Radix::sort 6-bits\n";
    // Test on different sizes
    for (int p = 8; p < 29; ++p)
    {
        const int SIZE = 1 << p;
        int* a = new int[SIZE];
        int* b = new int[SIZE];

        float elapsedTimes[4] = { 0.f, 0.f, 0.f, 0.f};

        genArray(SIZE - 1, a, 50);  // Leave a 0 at the end to test that edge case
        a[SIZE - 1] = 0;

        for (int i = 0; i < repeat; ++i)
        {
            zeroArray(SIZE, b);
            StreamCompaction::CPU::sort(SIZE, b, a);
            elapsedTimes[0] += StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation();

            zeroArray(SIZE, b);
            StreamCompaction::Radix::sort(SIZE, 31, b, a);
            elapsedTimes[1] += StreamCompaction::Radix::timer().getGpuElapsedTimeForPreviousOperation();

            zeroArray(SIZE, b);
            StreamCompaction::Radix::sort(SIZE, 15, b, a);
            elapsedTimes[2] += StreamCompaction::Radix::timer().getGpuElapsedTimeForPreviousOperation();

            zeroArray(SIZE, b);
            StreamCompaction::Radix::sort(SIZE, 6, b, a);
            elapsedTimes[3] += StreamCompaction::Radix::timer().getGpuElapsedTimeForPreviousOperation();
        }

        std::cout << std::setw(2) << p << "\t";
        std::cout << std::setw(19) << elapsedTimes[0] / repeat << "\t";
        std::cout << std::setw(19) << elapsedTimes[1] / repeat << "\t";
        std::cout << std::setw(19) << elapsedTimes[2] / repeat << "\t";
        std::cout << std::setw(19) << elapsedTimes[3] / repeat << "\n";

        delete[] a;
        delete[] b;
    }
}

int main(int argc, char * argv[])
{
    constexpr int repeat = 10;  // This is very time-consuming
    benchmark_scan_powerOfTwo(repeat);
    benchmark_scan_nonPowerOfTwo(repeat);
    benchmark_stream_compaction(repeat);
    benchmark_radix_sort(repeat);

    return 0;
}
#else
const int SIZE = 1 << 24; // feel free to change the size of array
const int NPOT = SIZE - 3; // Non-Power-Of-Two
int *a = new int[SIZE];
int *b = new int[SIZE];
int *c = new int[SIZE];

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
    printElapsedTime(StreamCompaction::Naive::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
    //printArray(SIZE, c, true);
    printCmpResult(NPOT, b, c);

    zeroArray(SIZE, c);
    printDesc("work-efficient scan, power-of-two");
    StreamCompaction::Efficient::scan(SIZE, c, a);
    printElapsedTime(StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
    //printArray(SIZE, c, true);
    printCmpResult(SIZE, b, c);

    zeroArray(SIZE, c);
    printDesc("work-efficient scan, non-power-of-two");
    StreamCompaction::Efficient::scan(NPOT, c, a);
    printElapsedTime(StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
    //printArray(NPOT, c, true);
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
    //printArray(count, c, true);
    printCmpLenResult(count, expectedCount, b, c);

    zeroArray(SIZE, c);
    printDesc("work-efficient compact, non-power-of-two");
    count = StreamCompaction::Efficient::compact(NPOT, c, a);
    printElapsedTime(StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
    //printArray(count, c, true);
    printCmpLenResult(count, expectedNPOT, b, c);

    zeroArray(SIZE, c);
    printDesc("thrust::remove_if compact, power-of-two");
    count = StreamCompaction::Thrust::compact(SIZE, c, a);
    printElapsedTime(StreamCompaction::Thrust::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
    //printArray(count, c, true);
    printCmpLenResult(count, expectedCount, b, c);

    zeroArray(SIZE, c);
    printDesc("thrust::remove_if compact, non-power-of-two");
    count = StreamCompaction::Thrust::compact(NPOT, c, a);
    printElapsedTime(StreamCompaction::Thrust::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
    //printArray(count, c, true);
    printCmpLenResult(count, expectedNPOT, b, c);

    // Sort tests
    printf("\n");
    printf("*****************************\n");
    printf("****** RADIX SORT TESTS *****\n");
    printf("*****************************\n");

    genArray(SIZE - 1, a, 50);  // Leave a 0 at the end to test that edge case
    a[SIZE - 1] = 0;
    printArray(SIZE, a, true);

    zeroArray(SIZE, b);
    printDesc("std::stable_sort, power-of-two, 6 bits");
    StreamCompaction::CPU::sort(SIZE, b, a);
    printElapsedTime(StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation(), "(std::chrono Measured)");
    printArray(SIZE, b, true);

    zeroArray(SIZE, c);
    printDesc("radix sort, power-of-two, 6 bits");
    StreamCompaction::Radix::sort(SIZE, 6, c, a);
    printElapsedTime(StreamCompaction::Radix::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
    //printArray(SIZE, c, true);
    printCmpResult(SIZE, b, c);

    zeroArray(SIZE, b);
    printDesc("std::stable_sort, non-power-of-two, 6 bits");
    StreamCompaction::CPU::sort(NPOT, b, a);
    printElapsedTime(StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation(), "(std::chrono Measured)");
    printArray(NPOT, b, true);

    zeroArray(SIZE, c);
    printDesc("radix sort, non-power-of-two, 6 bits");
    StreamCompaction::Radix::sort(NPOT, 6, c, a);
    printElapsedTime(StreamCompaction::Radix::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
    //printArray(NPOT, c, true);
    printCmpResult(SIZE, b, c);

#ifdef _WIN32
    system("pause"); // stop Win32 console from closing on exit
#endif

    delete[] a;
    delete[] b;
    delete[] c;
}
#endif

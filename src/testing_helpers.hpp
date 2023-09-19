#pragma once

#include <cstdlib>
#include <cstdio>
#include <iostream>
#include <string>
#include <ctime>

#define DEBUG 1
#define PRINT_ARRAY 1

template<typename T>
int cmpArrays(int n, T *a, T *b) {
#if DEBUG
    for (int i = 0; i < n; i++) {
        if (a[i] != b[i]) {
            printf("    a[%d] = %d, b[%d] = %d\n", i, a[i], i, b[i]);
            return 1;
        }
    }
#endif
    return 0;
}

void printDesc(const char *desc) {
#if DEBUG
    printf("==== %s ====\n", desc);
#endif
}

template<typename T>
void printCmpResult(int n, T *a, T *b) {
#if DEBUG
    printf("    %s \n",
            cmpArrays(n, a, b) ? "FAIL VALUE" : "passed");
#endif
}

template<typename T>
void printCmpLenResult(int n, int expN, T *a, T *b) {
#if DEBUG
    if (n != expN) {
        printf("    expected %d elements, got %d\n", expN, n);
    }
    printf("    %s \n",
            (n == -1 || n != expN) ? "FAIL COUNT" :
            cmpArrays(n, a, b) ? "FAIL VALUE" : "passed");
#endif
}

void zeroArray(int n, int *a) {
    for (int i = 0; i < n; i++) {
        a[i] = 0;
    }
}

void onesArray(int n, int *a) {
    for (int i = 0; i < n; i++) {
        a[i] = 1;
    }
}

void genArray(int n, int *a, int maxval) {
    srand(time(nullptr));

    for (int i = 0; i < n; i++) {
        a[i] = rand() % maxval;
    }
}

void printArray(int n, int *a, bool abridged = false) {
#if DEBUG && PRINT_ARRAY
    printf("    [ ");
    for (int i = 0; i < n; i++) {
        if (abridged && i + 2 == 15 && n > 16) {
            i = n - 2;
            printf("... ");
        }
        printf("%3d ", a[i]);
    }
    printf("]\n");
#endif
}

template<typename T>
void printElapsedTime(T time, std::string note = "")
{
#if DEBUG
    std::cout << "   elapsed time: " << time << "ms    " << note << std::endl;
#endif
}

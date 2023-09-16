CUDA Stream Compaction
======================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 2**

* Xinyu Niu
  * [personal website](https://xinyuniu6.wixsite.com/my-site-1)
* Tested on: Windows 11, i9-13980HX @ 2.20GHz 16GB, RTX 4070 16185MB (Personal)


## Introduction

This project focuses on implementing *Scan* (*Prefix Sum*) and *Stream Compaction* algorithms in CUDA. Scan algorithms are about doing prefix sum on an array, and Stream Compaction algorithm is about removing elements that meet some given conditions from an array. In this project, the stream compaction implementations will remove `0`s from an array of `int`s.

## Implemented Features

In this project, I completed the following features:

* CPU Scan & Stream Compaction
* Naive GPU Scan Algorithm
* Work-Efficient GPU Scan & Stream Compaction
* GPU Scan using Thrust
* Optimized Work-Efficient GPU Scan(Extra Credit)
* Radix Sort (Extra Credit)

## Performance Analysis
## Output
```
****************
** SCAN TESTS **
****************
    [  18  18  46  13  26  17  26   3   8  27  14  46   5 ...  33   0 ]
==== cpu scan, power-of-two ====
   elapsed time: 0ms    (std::chrono Measured)
    [   0  18  36  82  95 121 138 164 167 175 202 216 262 ... 410791936 410791969 ]
==== cpu scan, non-power-of-two ====
   elapsed time: 0ms    (std::chrono Measured)
    [   0  18  36  82  95 121 138 164 167 175 202 216 262 ... 410791854 410791889 ]
    passed
==== naive scan, power-of-two ====
   elapsed time: 17.4608ms    (CUDA Measured)
    passed
==== naive scan, non-power-of-two ====
   elapsed time: 17.1901ms    (CUDA Measured)
    passed
==== work-efficient scan, power-of-two ====
   elapsed time: 6.80093ms    (CUDA Measured)
    passed
==== work-efficient scan, non-power-of-two ====
   elapsed time: 6.6815ms    (CUDA Measured)
    passed
==== thrust scan, power-of-two ====
   elapsed time: 1.46637ms    (CUDA Measured)
    passed
==== thrust scan, non-power-of-two ====
   elapsed time: 1.10493ms    (CUDA Measured)
    passed

*****************************
** STREAM COMPACTION TESTS **
*****************************
    [   0   2   2   3   0   3   2   3   2   3   2   0   3 ...   3   0 ]
==== cpu compact without scan, power-of-two ====
   elapsed time: 32.9866ms    (std::chrono Measured)
    [   2   2   3   3   2   3   2   3   2   3   2   2   1 ...   2   3 ]
    passed
==== cpu compact without scan, non-power-of-two ====
   elapsed time: 32.8308ms    (std::chrono Measured)
    [   2   2   3   3   2   3   2   3   2   3   2   2   1 ...   3   1 ]
    passed
==== cpu compact with scan ====
   elapsed time: 60.2923ms    (std::chrono Measured)
    [   2   2   3   3   2   3   2   3   2   3   2   2   1 ...   2   3 ]
    passed
==== work-efficient compact, power-of-two ====
   elapsed time: 11.0502ms    (CUDA Measured)
    passed
==== work-efficient compact, non-power-of-two ====
   elapsed time: 11.5582ms    (CUDA Measured)
    passed

```
## Extra Credit
**1. Optimized GPU Efficient Scan**

I attempted to optimize the performance by adjusting blocks launched at during the loop for upper sweep and down sweep. During upper sweep, each iteration the block number shrinks to half. During down sweep, each iteration the block number expands to twice.

**2. Radix Sort**

I have implemented the Radix Sort algorithm, which can be called using ```StreamCompaction::Efficient::radixSort()```.

The below output resulted from comparing the sorted arry using my implementation and std::sort.

```
*****************************
** RADIX SORT TESTS **
*****************************
    [   0  17   0  16  17   5  18   9   5  18  19  18  10 ...   3   0 ]
==== Radix Sort ====
   elapsed time: 8.7022ms    (std::chrono Measured)
    [   0   0   0   0   0   0   0   0   0   0   0   0   0 ...  19  19 ]
    passed
```
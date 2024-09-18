CUDA Stream Compaction
======================

> ***University of Pennsylvania, CIS 5650: GPU Programming and Architecture, Project 2***
> * Michael Mason
>   + [Personal Website](https://www.michaelmason.xyz/), [LinkedIn](https://www.linkedin.com/in/mikeymason/)
> * Tested on: Windows 11, Ryzen 9 5900HS @ 3.00GHz 16GB, RTX 3080 (Laptop) 8192MB

In this project, **I implemented a GPU stream compaction algorithm using CUDA**, with the goal of removing zeros from an array of integers. 

Stream compaction is a critical technique in GPU programming, widely applicable in various scenarios (i.e. path tracing where terminated rays are compacted out of the working set).

This project involved both CPU and GPU implementations of the *all-prefix-sums* (or simply *scan*) algorithm as a foundation for the compaction process.

I used [Chapter 39 from GPU Gems 3 ](https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda)as the basis for the GPU implementations.

Three implementations of the *scan* algorithm were used for experimentation: 

1. The CPU implementation of the scan algorithm.
2. The "Naive" GPU implementation of the scan algorithm, as presented in *"Data Parallel Algorithms" by Hillis and Steele (1986)*
3. A "Work-Efficient" GPU implementation of the scan algorithm, as presented in *"Prefix Sums and Their Applications." by Belloch (1990)*

## ✅ Extra Credit Features

> ❔  ***Note to outside reader:*** *check instructions.md for details.*

* ***(For Part 5)*** My Work-Efficient GPU implementation of scan runs faster than the CPU approach (at larger array sizes). See the [Results](#results) and [Output](#-output) sections below. 

* ***(For Part 6.2)*** My work-Efficient GPU scan implementation uses shared memory and an arbitrary number blocks. However, there are some caveats: 
  + This was done only for the Work-efficient version. The naive implementation still uses global memory.  
  + I made no efforts to optimize for bank conflicts or occupancy.  

## ⏲️ Performance Analysis

### Results

#### Figure 1: Scan Implementations
![Scan Implementations - Array Size vs Time - Lower is Better](https://github.com/user-attachments/assets/b0f1c988-588c-4e85-b109-0b223a2ec39f)

#### Figure 2: Stream Compaction Implementations
![Stream Compact Implementations - Array Size vs Time - Lower is Better](https://github.com/user-attachments/assets/02a4c290-1069-40b5-9458-b3aaf45b5c1d)

### Explanation

> TODO

## 📃 Output 
```
***** Array size: 33554432 (1 << 25) *****
****************
** SCAN TESTS **
****************
    [  21  40  46   1  13  25  19   6  38  33  45   7  22 ...  37   5 ]
==== cpu scan, power-of-two ====
   elapsed time: 19.1649ms    (std::chrono Measured)
    [   0  21  61 107 108 121 146 165 171 209 242 287 294 ... 821704263 821704300 ]
==== cpu scan, non-power-of-two ====
   elapsed time: 19.9034ms    (std::chrono Measured)
    passed
==== naive scan, power-of-two ====
   elapsed time: 22.5557ms    (CUDA Measured)
    passed
==== naive scan, non-power-of-two ====
   elapsed time: 22.1012ms    (CUDA Measured)
    passed
==== work-efficient scan, power-of-two ====
   elapsed time: 11.38ms    (CUDA Measured)
    passed
==== work-efficient scan, non-power-of-two ====
   elapsed time: 11.1973ms    (CUDA Measured)
    passed
==== thrust scan, power-of-two ====
   elapsed time: 1.36806ms    (CUDA Measured)
    passed
==== thrust scan, non-power-of-two ====
   elapsed time: 1.49094ms    (CUDA Measured)
    passed

*****************************
** STREAM COMPACTION TESTS **
*****************************
    [   3   1   2   2   0   0   0   2   2   1   1   3   3 ...   3   0 ]
==== cpu compact without scan, power-of-two ====
   elapsed time: 58.7264ms    (std::chrono Measured)
    [   3   1   2   2   2   2   1   1   3   3   1   1   1 ...   3   3 ]
    passed
==== cpu compact without scan, non-power-of-two ====
   elapsed time: 58.5103ms    (std::chrono Measured)
    [   3   1   2   2   2   2   1   1   3   3   1   1   1 ...   2   3 ]
    passed
==== cpu compact with scan ====
   elapsed time: 103.895ms    (std::chrono Measured)
    [   3   1   2   2   2   2   1   1   3   3   1   1   1 ...   3   3 ]
    passed
==== work-efficient compact, power-of-two ====
   elapsed time: 11.4627ms    (CUDA Measured)
    [   3   1   2   2   2   2   1   1   3   3   1   1   1 ...   3   3 ]
    passed
==== work-efficient compact, non-power-of-two ====
   elapsed time: 14.8408ms    (CUDA Measured)
    [   3   1   2   2   2   2   1   1   3   3   1   1   1 ...   2   3 ]
    passed
```

[^1]: hello 

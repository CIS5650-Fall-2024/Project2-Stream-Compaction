CUDA Stream Compaction
======================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 2**

* Yuhan Liu
  * [LinkedIn](https://www.linkedin.com/in/yuhan-liu-), [personal website](https://liuyuhan.me/), [twitter](https://x.com/yuhanl_?lang=en), etc.
* Tested on: Windows 11 Pro, Ultra 7 155H @ 1.40 GHz 32GB, RTX 4060 8192MB (Personal Laptop)

## README!

### Project Description

* This project investigates parallel algorithms by implementing various versions of scan (prefix-sum), stream compaction (remove unwated elements), and sort algorithms. We begin with CPU implementations of all the aforementioned algorithms before developing GPU versions, including naive and work-efficient approaches, for comparison purposes. 

### Performance Analysis

**Optimizing Block Size**

<img src="https://github.com/yuhanliu-tech/GPU-Stream-Compaction/blob/main/img/block_opt.png" width="600"/>

* To decide on one block size for all parallel algorithm implementations, I assessed the naive and work-efficient runtimes with increasing block sizes. In the previous project, we established a tradeoff concerning block size between increasing GPU utilization and limiting resource availability for each block. Thus, we can choose an optimal block size by finding the dip in runtimes, which in this case (although close) we choose and proceed with a block size of 256. 

#### Comparison of GPU Scan Implementations

| CPU |  Naive  |   Work-Efficient  | Thrust |
| :------------------------------: |:------------------------------: |:-----------------------------------------------: |:-----------------------------------------------:|
| For smaller datasets, the overhead of managing parallel execution on a GPU (e.g., kernel launches, thread synchronization) offsets the parallel advantage, making the CPU more efficient. However, as data scales, the complexity of scanning on the CPU increases linearly. In the graph below where array sizes increase exponentially, the complexity of CPU scanning does so as well. | The naive parallel approach rivals the work-efficient approach for most of the array sizes below. However, this method has a memory latency bottleneck: its excessive use of global memory in the summation causes the runtime to swell as the array size grows. | The work-efficient GPU scan starts with more overhead than the naive implementation for smaller arrays, but as they grow larger, it outperforms both the CPU and naive versions. The work-efficient binary tree structure reduces unnecessary memory operations and optimizes thread usage by performing operations in place. The bottleneck here is thread usage: the binary tree structure leaves threads underutilized as the number of operations at each step in the algorithm reduces. Because of this, more primitive methods like naive and CPU come close in runtime for smaller array sizes. | Referencing the NVIDIA developer [documentation](https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda) for exclusive scan, we notice that the Thrust implementation employs techniques to improve performance, including optimizing shared memory and load balancing. Setting up these optimizations, like bankconflict avoidance and loop unrolling, introduces overhead, which make more primitive methods more efficient for small array sizes. However, even in the graph below, the thrust scan implementation clearly does not scale as significantly as any of the other scan implementations.        |

<img src="https://github.com/yuhanliu-tech/GPU-Stream-Compaction/blob/main/img/scan_perf.png" width="600"/>

#### Output of Testing 
test array SIZE: 2^18, blockSize: 256

```
****************
** SCAN TESTS **
****************
    [  35  49  36  24   1  48  46   6  49  24  44  47   5 ...   2   0 ]
==== cpu scan, power-of-two ====
   elapsed time: 0.4305ms    (std::chrono Measured)
    [   0  35  84 120 144 145 193 239 245 294 318 362 409 ... 6412415 6412417 ]
==== cpu scan, non-power-of-two ====
   elapsed time: 0.5392ms    (std::chrono Measured)
    [   0  35  84 120 144 145 193 239 245 294 318 362 409 ... 6412308 6412343 ]
    passed
==== naive scan, power-of-two ====
   elapsed time: 0.368768ms    (CUDA Measured)
    passed
==== naive scan, non-power-of-two ====
   elapsed time: 0.219936ms    (CUDA Measured)
    passed
==== work-efficient scan, power-of-two ====
   elapsed time: 0.333856ms    (CUDA Measured)
    passed
==== work-efficient scan, non-power-of-two ====
   elapsed time: 0.366688ms    (CUDA Measured)
    passed
==== thrust scan, power-of-two ====
   elapsed time: 1.55728ms    (CUDA Measured)
    passed
==== thrust scan, non-power-of-two ====
   elapsed time: 0.344288ms    (CUDA Measured)
    passed

*****************************
** STREAM COMPACTION TESTS **
*****************************
    [   1   3   0   0   3   2   0   2   1   2   0   1   3 ...   2   1 ]
==== cpu compact without scan, power-of-two ====
   elapsed time: 0.6932ms    (std::chrono Measured)
    [   1   3   3   2   2   1   2   1   3   3   2   2   3 ...   2   1 ]
    passed
==== cpu compact without scan, non-power-of-two ====
   elapsed time: 0.7058ms    (std::chrono Measured)
    [   1   3   3   2   2   1   2   1   3   3   2   2   3 ...   1   1 ]
    passed
==== cpu compact with scan ====
   elapsed time: 1.3968ms    (std::chrono Measured)
    [   1   3   3   2   2   1   2   1   3   3   2   2   3 ...   2   1 ]
    passed
==== work-efficient compact, power-of-two ====
   elapsed time: 0.445152ms    (CUDA Measured)
    passed
==== work-efficient compact, non-power-of-two ====
   elapsed time: 0.665728ms    (CUDA Measured)
    passed

*****************************
***** RADIX SORT TESTS ******
*****************************

Radix Sort, hard-coded lecture example test

    [   4   7   2   6   3   5   1   0 ]
==== cpu sort (std::sort) ====
   elapsed time: 0.0002ms    (std::chrono Measured)
    [   0   1   2   3   4   5   6   7 ]
==== radix sort ====
   elapsed time: 5.5552ms    (CUDA Measured)
    [   0   1   2   3   4   5   6   7 ]
    passed

Radix Sort, pow2 consecutive ints

    [   0   1   2   3   4   5   6   7   8   9  10  11  12 ... 262142 262143 ]
==== cpu sort (std::sort) ====
   elapsed time: 1.4753ms    (std::chrono Measured)
    [   0   1   2   3   4   5   6   7   8   9  10  11  12 ... 262142 262143 ]
==== radix sort ====
   elapsed time: 13.777ms    (CUDA Measured)
    [   0   1   2   3   4   5   6   7   8   9  10  11  12 ... 262142 262143 ]
    passed

Radix Sort, non-pow2 consecutive ints

    [   0   1   2   3   4   5   6   7   8   9  10  11  12 ... 262139 262140 ]
==== cpu sort (std::sort) ====
   elapsed time: 1.3184ms    (std::chrono Measured)
    [   0   1   2   3   4   5   6   7   8   9  10  11  12 ... 262139 262140 ]
==== radix sort ====
   elapsed time: 13.37ms    (CUDA Measured)
    [   0   1   2   3   4   5   6   7   8   9  10  11  12 ... 262139 262140 ]
    passed

Radix Sort, non-pow2 shuffled ints

    [ 4785 499 16736 27824 14951 31998 8696 23806 7549 30974 31344 14697 29955 ... 21785 7497 ]
==== cpu sort (std::sort) ====
   elapsed time: 14.9359ms    (std::chrono Measured)
    [   0   0   0   0   0   0   0   0   1   1   1   1   2 ... 32767 32767 ]
==== radix sort ====
   elapsed time: 18.8853ms    (CUDA Measured)
    [   0   0   0   0   0   0   0   0   1   1   1   1   2 ... 32767 32767 ]
    passed
```

### Additional Feature: Radix Sort

* I implemented parallel radix sort as an additional module to the `stream_compaction` subproject (and additionally, a CPU std::sort function for comparison). The implementation can be found in `radix.cu` and `radix.h`. 

* The radix sort function takes as input, the size of the array, along with pointers to the input and output arrays. Below is a code snippet from a radix test in the main method, showing how it is called:

```
StreamCompaction::Radix::sort(SIZE, c, a); // a is the array to be sorted, which is already-existing
printArray(SIZE, c, true); // print the output sorted array, which is saved in c
```

* I wrote several tests for radix sort, in which I compare both the correctness and runtime of my implementation to the CPU equivalent (std::sort). First, I compared radix sort with CPU sort on the fixed array [4, 7, 2, 6, 3, 5, 1, 0] from lecture. Then, I evaluated radix sort on sequential arrays of size power-of-two and non-power-of-two sizes. Lastly, I tested radix sort on a shuffled arrays. For each test, I compared the results and performance of radix sort with CPU sort, measuring CUDA performance where applicable.

**Radix sort Performance Evaluation**
* Radix sort performs worse than std::sort for smaller array sizes due to overhead and initialization costs associated with GPU processing and memory. However, as the array size grows, radix sort's performance improves relative to std::sort. This makes sense with the performance graph as radix sort has a complexity of O(n⋅k), where k is the number of bits, while standard CPU sort has a complexity of O(n*log(n)).

<img src="https://github.com/yuhanliu-tech/GPU-Stream-Compaction/blob/main/img/radix_perf.png" width="600"/>

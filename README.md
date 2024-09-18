CUDA Stream Compaction
======================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 2**

* Jichu Mao
  * [LinkedIn](https://www.linkedin.com/in/jichu-mao-a3a980226/)
  *  [Personal Website](https://jichu.art/)
* Tested on: Windows 11,  i7-13700K @ 3.40 GHz, 32GB, RTX 4090 24GB

## Overview
In this project, I implemented GPU-based stream compaction and scan algorithms using CUDA.
Stream compaction is a critical operation in GPU programming, particularly for applications like path tracing where it's necessary to efficiently remove inactive elements (e.g., terminated rays) from large datasets.

The primary goal was to implement several versions of the scan (prefix sum) algorithm and use them to perform stream compaction:

 * **CPU Scan and Stream Compaction**: Baseline implementations for correctness verification.
 * **Naive GPU Scan**: A straightforward GPU implementation using global memory.
 * **Work-Efficient GPU Scan and Stream Compaction**: An optimized GPU implementation that reduces redundant computations.
 * **Thrust Scan**: Utilizing NVIDIA's Thrust library for comparison.
 * **Optimizations**(Extra): Investigated and optimized performance bottlenecks in the GPU implementations.


## Descripitions 

### CPU Scan and Stream Compaction
 * Exclusive Scan: Implemented a CPU version of the exclusive prefix sum using a simple for-loop.
 * Stream Compaction without Scan: Removed zero elements from an array without using scan.
 * Stream Compaction with Scan: Used the scan result to efficiently compact an array by mapping, scanning, and scattering.



### Naive GPU Scan Algorithm
 * Implemented the naive scan algorithm on the GPU based on a straightforward parallel prefix sum approach.
 * Used global memory and multiple kernel launches for each step of the algorithm.
 * Handled non-power-of-two array sizes by padding the input array.

First do exclusive scan, then do shift right to get inclusive scan array.

![](img/naive.jpeg)

### Work-Efficient GPU Scan and Stream Compaction

Implemented the work-efficient scan algorithm using a balanced binary tree approach (up-sweep and down-sweep phases).

   
#### Step 1. Up-Sweep Phase
This builds a sum in a tree structure.
Starting from the leaves, each level of the tree computes partial sums of its two children and stores the result at the parent node. 
This process continues until the root contains the total sum of the array.

![](img/workeff1.jpeg)

#### Step 2. Down-Sweep Phase
This phase propagates partial sums back down the tree.
The total sum at the root is replaced with zero, and each parent passes its original value to its left child,
while the new value for the right child is the sum of the parent's original value and the left child’s value. This produces the final exclusive prefix sum.

![](img/workeff2.jpeg)

#### Step 3. Convert the exclusive scan to an inclusive scan
After conversion, we can output the results.

### Thrust Scan Implementation
Simply used the thrust::exclusive_scan(first, last, result) function from the Thrust library for performance comparison.

### Stream Compaction
Implemented GPU-based stream compaction using the work-efficient scan, including mapping to booleans and scattering.

## Performance Analysis
### Block Size Opimization for each Implementation

### CPU & GPU Scan Implementations Comparasion

### Bottlenecks

### Output Results
The following tests were ran on：
* Array size of **2<sup>27</sup>**
* A non-power-of-two array size of **2<sup>27</sup> - 3**
* A block size of **256**
* With thread reduction mode **on**.

```
****************
** SCAN TESTS **
****************
    [  43  21  19   5  12   0  25  24  36  44  25  38  44 ...  32   0 ]
==== cpu scan, power-of-two ====
   elapsed time: 190.116ms    (std::chrono Measured)
    [   0  43  64  83  88 100 100 125 149 185 229 254 292 ... -1007916206 -1007916174 ]
==== cpu scan, non-power-of-two ====
   elapsed time: 190.262ms    (std::chrono Measured)
    [   0  43  64  83  88 100 100 125 149 185 229 254 292 ... -1007916281 -1007916249 ]
    passed
==== naive scan, power-of-two ====
   elapsed time: 35.0255ms    (CUDA Measured)
    passed
==== naive scan, non-power-of-two ====
   elapsed time: 34.8509ms    (CUDA Measured)
    passed
==== work-efficient scan, power-of-two ====
   elapsed time: 26.2033ms    (CUDA Measured)
    passed
==== work-efficient scan, non-power-of-two ====
   elapsed time: 25.4444ms    (CUDA Measured)
    passed
==== thrust scan, power-of-two ====
   elapsed time: 2.90944ms    (CUDA Measured)
    passed
==== thrust scan, non-power-of-two ====
   elapsed time: 2.24038ms    (CUDA Measured)
    passed

*****************************
** STREAM COMPACTION TESTS **
*****************************
    [   1   0   3   2   2   1   3   0   1   0   2   0   3 ...   3   0 ]
==== cpu compact without scan, power-of-two ====
   elapsed time: 251.32ms    (std::chrono Measured)
    [   1   3   2   2   1   3   1   2   3   3   2   3   2 ...   2   3 ]
    passed
==== cpu compact without scan, non-power-of-two ====
   elapsed time: 246.755ms    (std::chrono Measured)
    [   1   3   2   2   1   3   1   2   3   3   2   3   2 ...   1   1 ]
    passed
==== cpu compact with scan ====
   elapsed time: 565.44ms    (std::chrono Measured)
    [   1   3   2   2   1   3   1   2   3   3   2   3   2 ...   2   3 ]
    passed
==== work-efficient compact, power-of-two ====
   elapsed time: 31.3884ms    (CUDA Measured)
    passed
==== work-efficient compact, non-power-of-two ====
   elapsed time: 30.1025ms    (CUDA Measured)
    passed
```

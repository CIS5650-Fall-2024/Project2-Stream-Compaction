CUDA Stream Compaction
======================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 2**

* Daniel Gerhardt
  * https://www.linkedin.com/in/daniel-gerhardt-bb012722b/
* Tested on: Windows 23H2, AMD Ryzen 9 7940HS @ 4GHz 32GB, RTX 4070 8 GB (Personal Laptop)

### Stream Compaction

## Description

Stream compaction is the process of taking an array and removing unwanted elements from it. In this project, that means removing 0s from an array. There are five different implementations being compared here. 

1. Compaction without scan on the CPU. This runs a simple O(n) for loop that checks if the input element is 0, and if it is not, adds it to the output.
2. Compaction with scan on the CPU. Scan is a "prefix sum" algorithm that outputs an array that contains at location i the sum of all elements up to element i. Scan can be used for compaction by creating an array of 1s and 0s that is parallel to the original input data, where a 1 represents the element at the same index is going to be in the output, and 0 represents the element is not going to be in the output. You can then accumulate the number of 1s to get an array that increases on the elements that should be contained in the output, and the value of the array at the element in the output is the final index of that element in the output.
3. Naive compaction on the GPU. The following figure shows what the approach looks like:
![](img/figure-39-2.jpg)
 By adding in parallel and only having a logarithmic number of iterations, this algorithm reduces the overall complexity to O(logn). But, there are O(nlogn) adds since in the worst case there are O(n) adds per iteration. The work efficient solutions seeks to reduce this factor.
4. Work efficient compaction on the GPU. The following figure shows what the latter stage of the approach looks like: ![](img/figure-39-4.jpg)
This is a much less intuitive approach. The algorithm involves 2 phases, the upsweep and the downsweep, pictured above. The upsweep is the same as the parallel reduction from method 3, except the algorithm occurs "in place" on the input data. Then, by treating the array as a tree and doing some clever summation, the amount of work can be reduced by filtering the sums down the "tree". This is done by setting the "root" -- the last element -- to zero, and then at each pass, giving each left child the parent's value, and setting the right child to the sum of the previous left child’s value and the parent's value. The upsweep has O(n) adds and the downsweep has O(n) adds and O(n) swaps, which reduces the complexity from method 3.
5. A wrapper for thrust's implementation of stream compaction for the sake of performance comparison.

# Sample Output

Run with 2^18 elements, this is what a sample program output looks like:

```
****************
** SCAN TESTS **
****************
    [  46   5  33  20   5  42  38  23  28  29   9  43  18 ...  43   0 ]
==== cpu scan, power-of-two ====
   elapsed time: 0.0677ms    (std::chrono Measured)
    [   0  46  51  84 104 109 151 189 212 240 269 278 321 ... 6416721 6416764 ]
==== cpu scan, non-power-of-two ====
   elapsed time: 0.063ms    (std::chrono Measured)
    [   0  46  51  84 104 109 151 189 212 240 269 278 321 ... 6416683 6416688 ]
    passed
==== naive scan, power-of-two ====
   elapsed time: 7.26304ms    (CUDA Measured)
    passed
==== naive scan, non-power-of-two ====
   elapsed time: 0.243776ms    (CUDA Measured)
    passed
==== work-efficient scan, power-of-two ====
   elapsed time: 0.407616ms    (CUDA Measured)
    passed
==== work-efficient scan, non-power-of-two ====
   elapsed time: 0.318944ms    (CUDA Measured)
    passed
==== thrust scan, power-of-two ====
   elapsed time: 0.083872ms    (CUDA Measured)
    passed
==== thrust scan, non-power-of-two ====
   elapsed time: 0.03472ms    (CUDA Measured)
    passed

*****************************
** STREAM COMPACTION TESTS **
*****************************
    [   0   1   1   2   1   2   0   1   0   3   3   1   2 ...   1   0 ]
==== cpu compact without scan, power-of-two ====
   elapsed time: 0.349ms    (std::chrono Measured)
    [   1   1   2   1   2   1   3   3   1   2   3   2   3 ...   2   1 ]
    passed
==== cpu compact without scan, non-power-of-two ====
   elapsed time: 0.3535ms    (std::chrono Measured)
    [   1   1   2   1   2   1   3   3   1   2   3   2   3 ...   3   1 ]
    passed
==== cpu compact with scan ====
   elapsed time: 0.7794ms    (std::chrono Measured)
    [   1   1   2   1   2   1   3   3   1   2   3   2   3 ...   2   1 ]
    passed
==== work-efficient compact, power-of-two ====
   elapsed time: 7.19795ms    (CUDA Measured)
    passed
==== work-efficient compact, non-power-of-two ====
   elapsed time: 0.545376ms    (CUDA Measured)
    passed
```

## Performance Analysis

The following data was collected using a block size of 128 running in release mode and utilizing cudaTimers and std::chrono to gather timing information. Memory swapping between the CPU and GPU was excluded where possible to focus the running time analysis on the algorithm. The size of 128 blocks was chosen from many tested as on my GPU it saw the best performance. Any references to non power of 2 or power of 2 algorithms refer to the input size used for the algorithm. This tests whether the approach is better or worse at handling array inputs where the input is a power of 2, or the input is not a power of 2. This is prevalent with the GPU approaches since the algorithms are tree based, which often rely on having a power of 2 length to properly touch all leaves.

# Charts

The following is a chart displaying how running time of the numerous methods changes with input size in the scan algorithm.

![](img/scan_graph.png)

The following is a chart displaying how running time of the numerous methods changes with input size in the compaction algorithm.

![](img/compaction_graph.png)

# Observations

In scan, the CPU dominated in performance over my implementations, but thrust proved to be the fastest at large array sizes, increasing much slower than other approaches. The CPU is marginally faster than thrust at small array sizes, because the parallelism is not fast enough to offset the heavier cost of transferring memory to and from the GPU. The thrust algorithm's supremacy shows that true harnessing of the GPU takes more than simply implementing an algorithm, but also smart usage of shared memory, data prefetching, and other strategies to utilize the parallel hardware. See more about the thrust algorithm at the bottom of the readme.

In stream compaction, the GPU has a much better performance at larger array sizes than the CPU. Note that the power of two and non power of two CPU runs without scan cover each other due to their negligible performance difference.

In scan, there is interesting behavior between the power of two and non power of two naive GPU algorithms. The non-power of two naive algorithm had an increasing difference in performance over the power of two input the larger the input became. This is likely because the overhead of padding zeroes increases as size increases. The same is not true of the compaction algorithms, likely because the main bottleneck of that algorithm is the global memory reads, which are present even if the input is a power of two. This was not present in the work efficient algorithm, which is so tight between the power of two and non power of two that the power of two line is not visible.

Across all algorithms and the multiple tests there is a very pleasing linear increase in performance. This is likely due to the running time being so closely tied to the array input size, and any additional overhead being trumped by the O(n) nature of the algorithmic approaches.

# Investigation of Thrust using NSight

The following is an image of the report from NSight.

![](img/nsight_thrust.png)

The green and red blocks are read and write from memory respectively. It appears they only happen once at the beginning and end of each test, since this image was taken from two consecutive thrust tests. That leads me to believe that the memory is loaded in to shared memory, and clever caching is used to avoid large amounts of usage from global memory, which is a large bottleneck in GPU programming.
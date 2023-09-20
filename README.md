CUDA Stream Compaction
======================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 2 - Stream Compaction**

* Tong Hu
  * [LinkedIn](https://www.linkedin.com/in/tong-hu-5819a122a/)
* Time tested on: Windows 11, Ryzen 7 1700X @ 3.4GHz 16GB, GTX 1080 16GB (Personal Computer)
* Nsight system analysis tested on: Windows 11, i5-11600K @ 3.91GHz 32GB, RTX 2060 6GB (Personal Desktop) (since GTX 1080 does not support GPU metric collection)

### Features
- CPU Scan & Stream Compaction
- Naive GPU Scan algorithm
- Work-Efficient GPU Scan (extra credit version) & Stream Compaction
- Using Thrust's Implementation

### Introduction
In this project, I implemented the scan function commonly used in stream compaction of int array. I implemented 3 different version (CPU version, naive parallel version, and efficient work version) of the Scan(exlusive Prefix Sum) algorithm, and use thrust's implemntation of exclusive scan, then compared the performance of each version of Scan.

### Roughly optimize the block sizes
Figure1. Time(ms) of scan function vs. block size (array size = $2^{27}$)
![Time(ms) vs. Block Size](/img/time_vs_blockSize.png)

Table 1. Time(ms) of scan function in different block size (array size = $2^{27}$)
| Block size      | 1       | 8       | 16      | 32      | 64      | 128     | 256     | 512     |
| --------------- | ------- | ------- | ------- | ------- | ------- | ------- | ------- | ------- |
| naïve pow2      | 4378.43 | 588.428 | 313.288 | 164.764 | 146.405 | 144.845 | 144.111 | 144.72  |
| naïve npow2     | 4373.24 | 571.921 | 291.82  | 164.399 | 146.764 | 144.936 | 144.916 | 144.649 |
| efficient pow2  | 328.222 | 64.9941 | 57.1655 | 58.7879 | 62.4402 | 62.0726 | 62.0135 | 60.9551 |
| efficient npow2 | 332.084 | 65.0913 | 56.9548 | 58.7963 | 62.0282 | 62.4519 | 61.8724 | 61.3648 |

From Figure 1 we can see that as the block size increases, the time of Scan function decreases rapidly, and after some point, the time does not change much although the number of block increased. From Table 1 we can see the naive parallel scan performs best when the block size equals to 256, while the work efficient parallel scan performs best when the block size is 16. Therefore I select 256 and 16 as block sizes for naive scan and work-efficient scan in the following performance analysis.

### Performance analysis
Figure 2. Time(ms) of scan function in diffrent array size.
![Time(ms) vs. array size](/img/time_vs_arraySize.png)

Table 2. Time(ms) of scan function in diffrent array size.
| log of array size | 4        | 8        | 16       | 20       | 22      | 24      | 25      | 26      | 27      | 28      | 29      | 30      |
| ----------------- | -------- | -------- | -------- | -------- | ------- | ------- | ------- | ------- | ------- | ------- | ------- | ------- |
| cpu               | 0.0002   | 0.0008   | 0.153    | 2.4054   | 9.0351  | 37.6997 | 78.1179 | 153.777 | 304.135 | 592.904 | 1266.01 | 2379.02 |
| naïve             | 0.02048  | 0.19456  | 0.318272 | 0.918432 | 3.64445 | 16.1397 | 33.5231 | 68.8091 | 144.143 | 302.976 | 639.805 | 4626.05 |
| efficient         | 0.306176 | 0.326656 | 0.566368 | 1.32307  | 2.16864 | 7.9192  | 15.282  | 29.2795 | 57.1435 | 113.729 | 234.424 | 429.09  |
| thrust            | 0.233504 | 0.131072 | 0.111616 | 0.929792 | 0.75264 | 1.57184 | 2.51597 | 3.45184 | 5.73133 | 10.7909 | 20.3489 | 112.254 |


From Figure 2 we can tell that when the array size increased, the time cost of Scan will increase. 

When the array size is small (smaller than $2^{16}$), CPU Scan performs better than GPU Scans and thrust's Scan. This is probably because the overhead of invoking GPU kernels overwhelms the benefit of parallel scans. When the array size is relative large, the time cost less for GPU Scans.

Comparing the naive parallel scan and work-efficient parallel scan, we can see from the figure that the work-efficient parallel scan performs better when the array size is large (greater than $2^{20}$). It takes naive parallel scan $O(n\log n)$ floating point adds operations while takes work-efficient scan $O(n)$ adds. Although both algorithm seems to run in parallel in the ideal case, in reality the number of threads run in parallel is bounded by hardware, and therefore, work-efficient scan performs better since the number of threads it need to lauch is smaller.

Figure 3. Nsight system trace
![Nsight system trace](/img/overall_2.png)

From Figure 3, we can see that compared with self-implemented Scan, the thrust's scan has very low DRAM bandwidth usage, and the unallocated warps in active SMs are also very low. Following are factors that might affect the performance:

1. Thrust's Scan might use shared memory, and memory coalescing when accessing global memory. This improves the memory throughput.
2. Thrust's Scan might optimize the block size and launch parameters based on workload and GPU type while our self-implemented scans hard code the block size only roughly optimized by eye.

### Bottleneck
Also, from Figure 3, we can know that the bottleneck of performance of Scan functions should be the memory bandwidth. The bandwidth of DRAM of both naive and work-efficient Scan are almost full.

### The output of the test program when the array size is $2^{25}$.
```
****************
** SCAN TESTS **
****************
    [  26   0  20  47  36   3  37  33  24  20  25  43  43 ...   6   0 ]
==== cpu scan, power-of-two ====
   elapsed time: 79.4345ms    (std::chrono Measured)
    [   0  26  26  46  93 129 132 169 202 226 246 271 314 ... 821748880 821748886 ]
==== cpu scan, non-power-of-two ====
   elapsed time: 78.2092ms    (std::chrono Measured)
    [   0  26  26  46  93 129 132 169 202 226 246 271 314 ... 821748831 821748870 ]
    passed
==== naive scan, power-of-two ====
   elapsed time: 34.0556ms    (CUDA Measured)
    passed
==== naive scan, non-power-of-two ====
   elapsed time: 33.1287ms    (CUDA Measured)
    passed
==== work-efficient scan, power-of-two ====
   elapsed time: 14.6806ms    (CUDA Measured)
    passed
==== work-efficient scan, non-power-of-two ====
   elapsed time: 14.6292ms    (CUDA Measured)
    passed
==== thrust scan, power-of-two ====
   elapsed time: 2.33472ms    (CUDA Measured)
    passed
==== thrust scan, non-power-of-two ====
   elapsed time: 2.2352ms    (CUDA Measured)
    passed

*****************************
** STREAM COMPACTION TESTS **
*****************************
    [   1   0   0   1   1   0   0   2   2   1   3   1   0 ...   3   0 ]
==== cpu compact without scan, power-of-two ====
   elapsed time: 82.3193ms    (std::chrono Measured)
    [   1   1   1   2   2   1   3   1   1   1   2   1   2 ...   2   3 ]
    passed
==== cpu compact without scan, non-power-of-two ====
   elapsed time: 82.2665ms    (std::chrono Measured)
    [   1   1   1   2   2   1   3   1   1   1   2   1   2 ...   3   2 ]
    passed
==== cpu compact with scan ====
   elapsed time: 329.982ms    (std::chrono Measured)
    [   1   1   1   2   2   1   3   1   1   1   2   1   2 ...   2   3 ]
    passed
==== work-efficient compact, power-of-two ====
   elapsed time: 21.669ms    (CUDA Measured)
    passed
==== work-efficient compact, non-power-of-two ====
   elapsed time: 21.466ms    (CUDA Measured)
    passed
```
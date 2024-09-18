CUDA Stream Compaction
======================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 2**

* Zhaojin Sun
  * www.linkedin.com/in/zjsun
* Tested on: Windows 11, i9-13900HX @ 2.2GHz 64GB, RTX 4090 Laptop 16GB

### 1. Features

**Functions implemented**
- CPU Scan & Stream Compaction
- Naive GPU Scan Algorithm
- Work-Efficient GPU Scan & Stream Compaction
- Thrust's Exclusive Scan Function
- [Extra Credit +10] Radix Sort Based on Work-Efficient Scan

**Optimizations made**
- [Extra Credit +5] Optimized number of threads used in work-efficient scan. 

### 2. Extra Credits
[Extra Credit +5] **Why is My GPU Approach So Slow?**

In the up-sweep and down-sweep phases, I optimized the process by shrinking the number of threads outside the kernel. 
I set the number of threads equal to the total number of steps, 
k. Inside each thread, I then expand these step indices back to the actual indices of 
k. This solution is thoroughly efficient because I completely avoided using the array length, 
n, to determine the number of blocks.

However, for arrays with very short lengths, the CPU will still outperform the GPU. This is because the basic unit of 
an up-sweep or down-sweep operation is inherently more complex than a simple short for-loop on the CPU. Nonetheless, this complexity becomes negligible as the array size increases significantly.

[Extra Credit +10] **Radix Sort Based on Work-Efficient Scan**

I have implemented radix sort algorithm by using the kernel of work-efficient scan. Here is its test results in release mode at length of 2^20, block size of 128:
```
*****************************
********* SORT TESTS ********
*****************************
==== radix sort, power-of-two ====
   elapsed time: 20.0816ms    (CUDA Measured)
    [   0   0   0   0   0   0   0   0   0   0   0   0   0 ...   3   3 ]
passed
==== radix sort, non-power-of-two ====
   elapsed time: 23.0973ms    (CUDA Measured)
    [   0   0   0   0   0   0   0   0   0   0   0   0   0 ...   3   3 ]
passed
Press any key to continue . . .
```


### 3. Performance Analysis
#### (i) Non-Power-of-Two Arrays
NPOT arrays has been a major issue that troubled me for days. However, I finally realized that the 
solution is simply padding an NPOT array with zeros to transform it into a POT array. These zeros are 
discarded after the up-sweep and down-sweep phases. This padding should only occur during the up-sweep and down-sweep 
stages since these are the only steps that require a balanced binary tree structure.

Padding with redundant zeros may impact performance, but this impact is minimal. Therefore, only POT arrays are used 
for the performance analysis below.




#### (ii) Selection of Block Size
Figure below shows in debug mode, under the length of 2^28, time cost by Naive and Work Efficient scan. From the graph we know that should pick the block size 
with the overall shortest scan time, which is 128, for further tests.
![scan_time_vs_block_size.png](img%2Fscan_time_vs_block_size.png)

Analysis:
- While this is a rough experiment, it provides valuable insight into the GPU’s performance when processing very large arrays.
- The figure also demonstrates that for long arrays, small block sizes are inefficient for the naive scan, resulting in 
wasted resources. However, as noted earlier, the work-efficient method has been optimized and handles small block sizes much more effectively.


#### (iii) Scan Result and Observation
Figure below shows in release mode, given fixed block size of 128, scan performance of all approaches. Both X and Y axis are log-scaled.
![scan_time_vs_array_size.png](img%2Fscan_time_vs_array_size.png)

Observation:
- CPU Scan: The time grows linearly with the array length, as expected, since the CPU processes each element sequentially.
- My GPU Scan: For large arrays, the advantages of parallel algorithms become apparent. The work-efficient scan is about 
ten times faster than the CPU version, though the naive method is slightly slower. However, the time complexity still grows 
nearly linearly, suggesting the current implementation hasn’t fully achieved its theoretical O(logn) potential.
- Thrust Scan: Thrust is highly optimized, exhibiting sub-linear time complexity. The performance gap between Thrust and
my implementation underscores the bottlenecks in my approach.


#### (iv) Compaction Result and Observation
Figure below shows in release mode, given fixed block size of 128, compaction performance of all approaches. Both X and Y axis are log-scaled.
![compaction_time_vs_array_size.png](img%2Fcompaction_time_vs_array_size.png)

Observation:
- During compaction, the work-efficient method requires two additional steps involving the kernels implemented in common.cu. 
This introduces some ```cudaMemcpy``` operations within the GPU timer, which slightly slows it down. Despite this, the work-efficient method still outperforms the CPU method when processing long arrays.


#### (v) Bottlenecks Analysis
For the CPU and naive scan methods, the bottlenecks lie in their respective computation approaches or similar inefficiencies 
found in the work-efficient scan. Therefore, I’ll focus on comparing my work-efficient scan with the highly optimized Thrust scan.

Below are a few analyses from Nsight Compute:
![nsight_overview.png](img%2Fnsight_overview.png)
![nsight_mine.png](img%2Fnsight_mine.png)
![nsight_thrust.png](img%2Fnsight_thrust.png)

Here are some possible bottlenecks I’ve identified:
- Memory I/O: One significant issue is the lack of shared memory in my implementation, while Thrust scan uses 6.16KB of 
static shared memory per block. Since shared memory is much faster than global memory, this is likely a key reason why Thrust outperforms my work-efficient scan.
- Computation Method: My work-efficient scan still contains inner loops with inefficient computations and poor memory 
throughput. Synchronization efficiency also seems suboptimal. Thrust appears to have some way to "squeeze" these loops, 
saving significant time by running much fewer cycles than my approach.
- Compute Throughput: Thrust has consistently higher compute throughput compared to my kernels. I suspect this is due to 
additional optimizations in kernel efficiency that Thrust implements.


I find Nsight and CUDA hardware concepts challenging to grasp, and my current understanding is fairly limited. 
I would greatly appreciate any feedback or corrections to help me improve.

### 4. Test Results
Below is the vanilla test result in release mode at length of 2^20, block size of 128.
```
****************
** SCAN TESTS **
****************
    [  23   6  22   7  41  17  30  28  20  32  46  40   9 ...   6   0 ]
==== cpu scan, power-of-two ====
   elapsed time: 401.431ms    (std::chrono Measured)
    [   0  23  29  51  58  99 116 146 174 194 226 272 312 ... -2015404609 -2015404603 ]
==== cpu scan, non-power-of-two ====
   elapsed time: 394.178ms    (std::chrono Measured)
    [   0  23  29  51  58  99 116 146 174 194 226 272 312 ... -2015404701 -2015404681 ]
    passed
==== naive scan, power-of-two ====
   elapsed time: 130.114ms    (CUDA Measured)
    passed
==== naive scan, non-power-of-two ====
   elapsed time: 129.647ms    (CUDA Measured)
    passed
==== work-efficient scan, power-of-two ====
   elapsed time: 45.0633ms    (CUDA Measured)
    passed
==== work-efficient scan, non-power-of-two ====
   elapsed time: 66.5025ms    (CUDA Measured)
    passed
==== thrust scan, power-of-two ====
   elapsed time: 5.39546ms    (CUDA Measured)
    passed
==== thrust scan, non-power-of-two ====
   elapsed time: 5.74288ms    (CUDA Measured)
    passed

*****************************
** STREAM COMPACTION TESTS **
*****************************
    [   3   0   1   3   3   3   3   0   3   1   3   0   0 ...   1   0 ]
==== cpu compact without scan, power-of-two ====
   elapsed time: 532.37ms    (std::chrono Measured)
    [   3   1   3   3   3   3   3   1   3   1   3   1   3 ...   2   1 ]
    passed
==== cpu compact without scan, non-power-of-two ====
   elapsed time: 523.009ms    (std::chrono Measured)
    [   3   1   3   3   3   3   3   1   3   1   3   1   3 ...   2   2 ]
    passed
==== cpu compact with scan ====
   elapsed time: 1271.93ms    (std::chrono Measured)
    [   3   1   3   3   3   3   3   1   3   1   3   1   3 ...   2   1 ]
    passed
==== work-efficient compact, power-of-two ====
   elapsed time: 81.0711ms    (CUDA Measured)
    passed
==== work-efficient compact, non-power-of-two ====
   elapsed time: 100.548ms    (CUDA Measured)
    passed
Press any key to continue . . .

```
It is very interesting that ```thrust::exclusive_scan(first, last, result)``` is way much faster in release mode than in debug mode.
In debug mode it is even slower than CPU benchmark.
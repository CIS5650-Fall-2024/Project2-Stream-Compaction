CUDA Stream Compaction
======================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 2**

* Christine Kneer
  * https://www.linkedin.com/in/christine-kneer/
  * https://www.christinekneer.com/
* Tested on: Windows 11, i7-13700HX @ 2.1GHz 32GB, RTX 4060 8GB (Personal Laptop)

# Part 1: Introduction

In this project, I implemented stream compaction algorithim that removes `0`s
from an array of `int`s. The stream compaction algorthim works in three steps:

- **Step 1**: Compute temporary array containing 1 if corresponding element meets criteria (non-zero), 0 if element does not meet criteria.

<p align="center">
 <img width="204" alt="image" src="https://github.com/user-attachments/assets/7f22acf3-bd35-4d04-8e41-0727034cf415">
</p>

- **Step 2**:  Run exclusive scan on temporary array, to calculate the positions where non-zero elements should be placed in the output array.
<p align="center">
  <img width="290" alt="image" src="https://github.com/user-attachments/assets/0c0ac9b5-5652-4eef-a023-d3de55cae2e9">
 </p>

- **Step 3**:  Scatter. Result of scan is index into final array, but we only write an element if temporary array has a 1.
  
<p align="center">
  <img width="299" alt="image" src="https://github.com/user-attachments/assets/8ca6c1a7-b085-4eeb-8f06-74a48c1c62b7">
 </p>


I have implementd a few different versions of Step 2, the *Scan* (*Prefix Sum*)
algorithm, namely the CPU version, the Naive GPU version, the Work-Efficient GPU version and the
Thrust library version. The *Scan* (*Prefix Sum*) algorithm plays a critical role in parallelizing
the stream compaction process, mainly in two ways:

- **Marking positions for non-zero elements**: The scan operation allows us to calculate the positions
  where non-zero elements should be placed in the output array. We can apply the scan to accumulate
  non-zero elements up to each index.
- **Efficient memory allocation**: Once the scan has been performed, the result tells us exactly how many
  elements will remain after compaction, allowing efficient memory usage.

## Part 1.1: CPU Scan

The CPU version is straightforward, working sequentially by iterating through the input array.
```
out[0] = 0;
for (int k = 1; k < n; ++k)
  out[k] = out[k – 1] + in[k - 1];
```

## Part 1.2: Naive GPU Scan

The Naive GPU scan uses the "Naive" algorithm from GPU Gems 3, Section 39.2.1. 

<p align="center">
<img width="362" alt="image" src="https://github.com/user-attachments/assets/f5514241-359d-453e-8517-d69f5c19623a">
</p>

The algorithm works by shifting the input array and performing element-wise additions in parallel, gradually
accumulating the sum. Although this method leverages parallelism, it has some inefficiencies in memory usage and requires `O(log n)` steps.

## Part 1.3: Work-Efficient GPU Scan

This uses the "Work-Efficient" algorithm from GPU Gems 3, Section 39.2.2, which uses a two-pass algorithm
that consists of an up-sweep(reduce) phase followed by a down-sweep (propagate) phase.

The **up-sweep** phase calculates partial sums by bulding a binary tree:

<p align="center">
<img width="204" alt="image" src="https://github.com/user-attachments/assets/3c24bfe0-86e9-4bab-832f-db1ece628eeb">
</p>

The **down-sweep** phase uses the results from the tree to propagate the final sums:

<p align="center">
<img width="284" alt="image" src="https://github.com/user-attachments/assets/56e40f67-5dfe-4fe5-8733-29f38af652a7">
</p>

This approach requires fewer operations and leads to better performance, achieving `O(n)` work complexity with efficient use of GPU resources.

## Part 1.4: Thrust Scan

In order to compare performance, I also used the Thrust library function `thrust::exclusive_scan(first, last, result)`.

# Part 2: Performance Analysis

In this section, we compare the performance between the difference scan algorithms, by measuring the actual execution time in ms. The GPU tests below all operate on a block size of **128**, which is tested to be the optimal block size, where smaller block size reduces performance and larger block size does not improve performance anymore.

## Part 2.1: Improving Work-Efficient GPU Scan

In the naive version, we launch the same number of threads per level, then terminate the thread if we determine that the thread is unneeded (using mod). 
```
for (int d = 0; d < numLevels; ++d) {
  upSweep <<<fullBlocksPerGrid, blockSize>>> (n, dev_idata, 1 << (d + 1));
}
```
```
__global__ void upSweep(int n, int* data, int step) {
  int index = threadIdx.x + (blockIdx.x * blockDim.x);
  if (index < n && (index % step == 0)) {
    data[index + step - 1] += data[index + (step >> 1) - 1];
 }
}
```

I optimized by dynamically launching as many thread as we need, reducing the number of lazy threads per level by a power of 2, which at the same time gets rid of the expensive mod expression. 

```
for (int d = 0; d < numLevels; ++d) {
    int numThreads = powerOf2Size / (1 << (d + 1));
    dim3 BlocksPerGrid((numThreads + blockSize - 1) / blockSize);
    upSweep <<<BlocksPerGrid, blockSize>>> (powerOf2Size, dev_idata, 1 << (d + 1));
}
```
```
__global__ void upSweep(int n, int* data, int step) {
  int index = threadIdx.x + (blockIdx.x * blockDim.x);
  if (index < n / step) {
    index *= step;
    data[index + step - 1] += data[index + (step >> 1) - 1];
  }
}
```
As seen from the chart below, the improved appraoch consistently acheives better performance (about x2).

|![improved](https://github.com/user-attachments/assets/a4e004d5-1320-4d23-9953-b7f51a7a23f5)|
|:--:|
|*block size = 128*|

## Part 2.2: General Comparative Analysis

|![cross](https://github.com/user-attachments/assets/4994f39f-4fd0-4b3c-8b45-56fe4bc34a23)|
|:--:|
|*block size = 128*|

We see that as a general trend, the execution time is as follows: `CPU > Naive > Work-Efficient > Thrust`. All of the algorithms are appraoching linear growth (though at different pace) as the array size gets larger and larger.

- **CPU Scan**: The CPU version is purely sequential. As the array size increases, the execution time growth dramatically, which is expected. The bottleneck is primarily **compuation**, as the sequential nature of the algorithm scales poorly.

    <p align="center">
  <img width="818" alt="image" src="https://github.com/user-attachments/assets/06d40843-0585-42a1-bb57-e5500a747128">
  </p>
  
- **Naive GPU Scan**: The Naive GPU version attempts to utilize parallelism, but involves unneccessary computation. **SM Warp Occupancy** is around 80%, which is not that bad, but **DRAM bandwidth** and **PCIe bandwidth** shows high peak, indicating excessive global memory access. The bottleneck is largely **memory I/O**.
  
<p align="center">
  <img width="607" alt="image" src="https://github.com/user-attachments/assets/41cfc28f-c196-4c26-b6e8-a335e32fc2f0">
 </p>
 
- **Work-Efficient GPU Scan**: The Work-Efficient GPU version improves on the Naive GPU approach by reducing the amount of unneccessary work and memory accesses. As a result, **DRAM bandwidth** is relatively lower, though peaks still occur during phases that involve global memory access. **SM Warp Occupancy** tends to decrease as the number of active threads (elements we process) per level decreases, which is inherent to the algorithm itself, but can be possibly improved through hybrid block sizes or thread consolidation. Overall, the bottleneck is **load balancing** and **memory I/O**.

<p align="center">
 <img width="716" alt="image" src="https://github.com/user-attachments/assets/f9619410-b659-4afb-a08e-c81bdb8e60d8">
  </p>

- **Thrust Scan**: The Thrust implementation shows very stable **SM Warp Occupancy**, consistent low **DRAM bandwidth** and **PCIe bandwidth**. It likely utilizes shared memory and registers in an efficient way.
  
  <p align="center">
<img width="761" alt="image" src="https://github.com/user-attachments/assets/467c3ce7-68ed-4ff9-b6ad-a1d15ba37ade">
  </p>

# Part 3: Application Output
```
****************
** SCAN TESTS **
****************
    [  46  21  19  33  23   4  20  20  35  40  32  17  24 ...   1   0 ]
==== cpu scan, power-of-two ====
   elapsed time: 1241.47ms    (std::chrono Measured)
    [   0  46  67  86 119 142 146 166 186 221 261 293 310 ... 263219201 263219202 ]
==== cpu scan, non-power-of-two ====
   elapsed time: 1266.65ms    (std::chrono Measured)
    [   0  46  67  86 119 142 146 166 186 221 261 293 310 ... 263219103 263219115 ]
    passed
==== naive scan, power-of-two ====
   elapsed time: 632.183ms    (CUDA Measured)
    passed
==== naive scan, non-power-of-two ====
   elapsed time: 628.37ms    (CUDA Measured)
    passed
==== work-efficient scan, power-of-two ====
   elapsed time: 190.335ms    (CUDA Measured)
    passed
==== work-efficient scan, non-power-of-two ====
   elapsed time: 190.565ms    (CUDA Measured)
    passed
==== thrust scan, power-of-two ====
   elapsed time: 20.0224ms    (CUDA Measured)
    passed
==== thrust scan, non-power-of-two ====
   elapsed time: 19.9178ms    (CUDA Measured)
    passed

*****************************
** STREAM COMPACTION TESTS **
*****************************
    [   0   2   0   3   3   1   3   3   0   1   0   0   0 ...   1   0 ]
==== cpu compact without scan, power-of-two ====
   elapsed time: 1438.67ms    (std::chrono Measured)
    [   2   3   3   1   3   3   1   1   1   1   3   1   1 ...   3   1 ]
    passed
==== cpu compact without scan, non-power-of-two ====
   elapsed time: 1380.96ms    (std::chrono Measured)
    [   2   3   3   1   3   3   1   1   1   1   3   1   1 ...   2   3 ]
    passed
==== cpu compact with scan ====
   elapsed time: 3860.88ms    (std::chrono Measured)
    [   2   3   3   1   3   3   1   1   1   1   3   1   1 ...   3   1 ]
    passed
==== work-efficient compact, power-of-two ====
   elapsed time: 261.587ms    (CUDA Measured)
    passed
==== work-efficient compact, non-power-of-two ====
   elapsed time: 261.878ms    (CUDA Measured)
    passed
```

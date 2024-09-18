CUDA Stream Compaction
======================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 2**

* Kyle Bauer
  * [LinkedIn](https://www.linkedin.com/in/kyle-bauer-75bb25171/), [twitter](https://x.com/KyleBauer414346)
* Tested on: Windows 10, i-7 12700 @ 2.1GHz 32GB, NVIDIA T1000 4GB (CETS Virtual Lab)

Analysis
---

<div align="center">
 <img src="img/Scan Implementation Comparison Pow2.svg" />
</div>

The CPU, Naive, and Work-Efficient implementations all scaled similarly with an increasing array size. Generally, doubling the array size would double the runtime of each algorithm.

The CPU and Work-Efficient implementations compared very similarly, with the Work-Efficient runtimes never straying more than 3% away from the CPU runtimes.

The Naive implemenation's runtime diverged slightly from the CPU and Work-Efficient runtimes at around the 2^21 array size mark. In runs with a lesser element size than this, Naive performed up to 6% faster (at 2^20 elements) compared to the CPU implementation. And in runs with a greater element size, Naive performed at most 10% worse (at 2^24 elements) than the CPU implementation.

The Thrust implementation is clearly the overall most performant option, pulling completely away from all other implementations as the array size increases.

<strong>Potential Bottlenecks:</strong>
1. Global Memory: Both the Naive and Work-Efficient algorithms were implemented using global memory with no shared memory, creating a massive amount of overhead anytime the implementations wish to read or write data.
2. Memory Locality: Both the Naive and Work-Efficient algorithms read and write data across very large arrays. As the algorithms progress, these memory accesses become progressively more sparse- randomly accessing the memory will cause cache thrashing decreasing the bus utilization.
3. GPU Utilization: The Naive algorithm suffers from not saturating the GPU (Many threads are ended early leaving a couple of active threads in a warp). This inherently decreases parallelism and will increase the runtime as the array size grows.

Sample Output
---

```
****************
** SCAN TESTS **
****************
    [  33  40  46  12  48  15   5  37  39  42  27  41  35 ...  10   0 ]
==== cpu scan, power-of-two ====
   elapsed time: 27.4656ms    (std::chrono Measured)
    [   0  33  73 119 131 179 194 199 236 275 317 344 385 ... 410928744 410928754 ]
==== cpu scan, non-power-of-two ====
   elapsed time: 26.8243ms    (std::chrono Measured)
    [   0  33  73 119 131 179 194 199 236 275 317 344 385 ... 410928700 410928722 ]
    passed
==== naive scan, power-of-two ====
   elapsed time: 31.6926ms    (CUDA Measured)
    passed
==== naive scan, non-power-of-two ====
   elapsed time: 30.7692ms    (CUDA Measured)
    passed
==== work-efficient scan, power-of-two ====
   elapsed time: 23.55ms    (CUDA Measured)
    passed
==== work-efficient scan, non-power-of-two ====
   elapsed time: 23.0375ms    (CUDA Measured)
    passed
==== thrust scan, power-of-two ====
   elapsed time: 1.71158ms    (CUDA Measured)
    passed
==== thrust scan, non-power-of-two ====
   elapsed time: 1.14893ms    (CUDA Measured)
    passed

*****************************
** STREAM COMPACTION TESTS **
*****************************
    [   0   3   2   0   2   0   0   2   0   0   3   3   2 ...   1   0 ]
==== cpu compact without scan, power-of-two ====
   elapsed time: 31.584ms    (std::chrono Measured)
    [   3   2   2   2   3   3   2   2   1   1   2   3   1 ...   1   1 ]
    passed
==== cpu compact without scan, non-power-of-two ====
   elapsed time: 35.5074ms    (std::chrono Measured)
    [   3   2   2   2   3   3   2   2   1   1   2   3   1 ...   2   2 ]
    passed
==== cpu compact with scan, power-of-two ====
   elapsed time: 74.7157ms    (std::chrono Measured)
    [   3   2   2   2   3   3   2   2   1   1   2   3   1 ...   1   1 ]
    passed
==== cpu compact with scan, non-power-of-two ====
   elapsed time: 73.4743ms    (std::chrono Measured)
    [   3   2   2   2   3   3   2   2   1   1   2   3   1 ...   2   2 ]
    passed
==== work-efficient compact, power-of-two ====
   elapsed time: 33.6798ms    (CUDA Measured)
    passed
==== work-efficient compact, non-power-of-two ====
   elapsed time: 24.5682ms    (CUDA Measured)
    passed
```

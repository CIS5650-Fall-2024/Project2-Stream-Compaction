CUDA Stream Compaction
======================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 2**

* Zixiao Wang
  * [LinkedIn](https://www.linkedin.com/in/zixiao-wang-826a5a255/)
* Tested on: Windows 11, i7-14700K @ 3.40 GHz 32GB, GTX 4070TI 12GB  4K Monitor (Personal PC)

### Background

* In this project, I implemented GPU stream compaction in CUDA, from scratch. This algorithm is widely used, and will be important for accelerating path tracer. I implemented several versions of the Scan (Prefix Sum) algorithm, a critical building block for stream compaction:
  * CPU Implementation: Begin by coding a CPU version of the Scan algorithm.
  * GPU Implementations:
     *  Naive Scan: Develop a straightforward GPU version that parallelizes the Scan operation.
     *  Work-Efficient Scan: Optimize GPU implementation to make better use of parallel resources, minimizing idle threads and redundant computations.
   
  * GPU Stream Compaction: Utilize Scan implementations to create a GPU-based stream compaction algorithm that efficiently removes zeros from an integer array.
* For more detail [GPU GEM](https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda).

### Prefix-Sum(Scan)

![](images/ScanSpeed.png)

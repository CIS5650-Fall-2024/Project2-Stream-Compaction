CUDA Stream Compaction
======================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 2**

* Yuhan Liu
  * [LinkedIn](https://www.linkedin.com/in/yuhan-liu-), [personal website](https://liuyuhan.me/), [twitter](https://x.com/yuhanl_?lang=en), etc.
* Tested on: Windows 11 Pro, Ultra 7 155H @ 1.40 GHz 32GB, RTX 4060 8192MB (Personal Laptop)

## README!

### Project Description

* XXXX

### Performance Analysis

**Optimizing Block Size**

<img src="https://github.com/yuhanliu-tech/GPU-Stream-Compaction/blob/main/img/block_opt.png" width="600"/>

#### Comparison of GPU Scan Implementations

| CPU |  Naive  |   Work-Efficient  | Thrust |
| :------------------------------: |:------------------------------: |:-----------------------------------------------: |:-----------------------------------------------:|
| xxxxx                            | xxxxxx                          |xxxxxx                                            |xxxxxx                                    |


<img src="https://github.com/yuhanliu-tech/GPU-Stream-Compaction/blob/main/img/scan_perf.png" width="600"/>

**Explaining of Phenomena in the Graph**

* Can you find the performance bottlenecks? Is it memory I/O? Computation? Is it different for each implementation?

#### Output of Testing 
(test array) SIZE: 2^19, blockSize: 128

```

```

#### Additional Feature: Radix Sort

* For radix sort, show how it is called and an example of its output.

* Additional tests for radix sort

* Performance Evaluation

<img src="https://github.com/yuhanliu-tech/GPU-Stream-Compaction/blob/main/img/radix_perf.png" width="600"/>

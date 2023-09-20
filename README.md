CUDA Stream Compaction
======================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 2**

* Xitong Zheng
  * [LinkedIn](https://www.linkedin.com/in/xitong-zheng-5b6543205/), [Instagram](https://www.instagram.com/simonz_zheng/), etc.
* Tested on: Windows 11, i7-12700k 32GB, GTX 4090 24GB

### Features Implemented
To make it clear, scan here refers to calculate the prefix sum of an array, whether exclusive or inclusive. Stream Compaction refers to remove the zero data in an array and compact the useful data together with less memory size required. This can be useful in spareMatrix and other fields.
- CPU Scan & Stream Compaction 
- Naive GPU Scan 
- Work-Efficent Algo GPU Scan & Stream Compaction
- Thrust Scan (for comparison)
- Optimized GPU Scan 
- Radix Sort
- GPU Scan Using Shared Memory 
#### Features to be included
- bank conflict relieve
- add Nsight Performance Analysis at readme
### Feature Details
### Performance Analysis
#### Find the optimized of each of implementation for minimal run time

#### Compare all of these GPU Scan implementations (Naive, Work-Efficient, and Thrust) to the serial CPU version of Scan. Plot a graph of the comparison (with array size on the independent axis).

#### Performance bottlenecks for each implementation.

#### Output of the test program


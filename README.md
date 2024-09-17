CUDA Stream Compaction
======================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 2**

* Mufeng Xu
  * [LinkedIn](https://www.linkedin.com/in/mufeng-xu/)
* Tested on: Windows 11, i9-13900H @ 2.6GHz 32GB, RTX 4080 Laptop 12282MB (Personal Computer)

> *GPU parallel algorithms have significantly better performance when the array sizes are large!*
> 
>![](img/benchmark-power-of-2(linear-y).png)
>![](img/benchmark-compaction-power-of-2(linear-y).png)

In this project, I implemented the following features:
- CPU Scan & Stream Compaction
- Naive GPU Scan Algorithm
- Work-Efficient GPU Scan & Stream Compaction
- Improved the performance of GPU scans with some index calculation tweaks.
- GPU Scan with `Thrust`
- GPU **Radix Sort**, and compared it with `std::stable_sort`.

## Performance Benchmarks

### Scan Algorithms

![](img/benchmark-power-of-2.png)

![](img/benchmark-non-power-of-2.png)

### Stream Compaction

![](img/benchmark-compaction-power-of-2.png)

### Radix Sort

![](img/benchmark-radix-sort.png)
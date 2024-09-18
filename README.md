CUDA Stream Compaction
======================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 2**

* Joanna Fisch
  * [LinkedIn](https://www.linkedin.com/in/joanna-fisch-bb2979186/), [Website](https://sites.google.com/view/joannafischsportfolio/home)
* Tested on: Windows 11, i7-12700H @ 2.30GHz 16GB, NVIDIA GeForce RTX 3060 (Laptop)

## Overview

This project implements GPU stream compaction using CUDA to remove zeros from an array of integers. Stream compaction is a crucial optimization technique, particularly useful for high-performance computing tasks like path tracing. By implementing various scan (prefix sum) algorithms, this project explores different strategies to leverage GPU parallelism efficiently.

## Features
#### 1. CPU Scan & Stream Compaction
* CPU Scan: Implements a simple exclusive prefix sum using a for loop.
* Compact Without Scan: A basic CPU method that removes zeros without relying on a scan operation.
* Compact With Scan: A more advanced method using scan to optimize the stream compaction process.

#### 2. Naive GPU Scan
* Implements a naive GPU scan algorithm based on the description in GPU Gems 3, Section 39.2.1. This implementation uses global memory and repeatedly swaps input/output arrays across several kernel launches.

#### 3. Work-Efficient GPU Scan & Stream Compaction
* Work-Efficient Scan: Implements a more optimized version using the tree-based approach from GPU Gems 3, Section 39.2.2.
* Stream Compaction Using Scan: Uses the work-efficient scan to perform stream compaction by mapping the input array to boolean values, scanning the array, and then scattering the valid elements.
* Handles non-power-of-two sized arrays efficiently.

#### 4. Thrust Library Integration
* Uses the Thrust library's exclusive_scan function to perform stream compaction with GPU-accelerated thrust primitives.

## Performance Analysis
#### Key Insights:
* Naive GPU Scan: While the naive approach is simple, it suffers from inefficiencies due to its multiple kernel launches and use of global memory.
* Work-Efficient GPU Scan: More performant than the naive version due to reduced memory bandwidth usage and in-place calculations.
* Thrust Implementation: Fast and highly optimized due to the underlying Thrust library's optimizations, but lacks flexibility for customization.

#### Bottlenecks Identified:
* Global Memory Access: Naive implementations suffer from excessive global memory reads/writes, leading to slower performance.
* Occupancy: The work-efficient algorithm shows reduced efficiency at deeper recursion levels due to reduced thread workload, limiting GPU occupancy.

CUDA Stream Compaction
======================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 2**

* Joanna Fisch
  * [LinkedIn](https://www.linkedin.com/in/joanna-fisch-bb2979186/), [Website](https://sites.google.com/view/joannafischsportfolio/home)
* Tested on: Windows 11, i7-12700H @ 2.30GHz 16GB, NVIDIA GeForce RTX 3060 (Laptop)

### Overview

This project implements GPU stream compaction using CUDA to remove zeros from an array of integers. Stream compaction is a crucial optimization technique, particularly useful for high-performance computing tasks like path tracing. By implementing various scan (prefix sum) algorithms, this project explores different strategies to leverage GPU parallelism efficiently.

### Features
1. CPU Scan & Stream Compaction
 * CPU Scan: Implements a simple exclusive prefix sum using a for loop.
 * Compact Without Scan: A basic CPU method that removes zeros without relying on a scan operation.
 * Compact With Scan: A more advanced method using scan to optimize the stream compaction process.


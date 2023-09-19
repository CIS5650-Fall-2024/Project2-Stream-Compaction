CUDA Stream Compaction
======================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 2**

* Utkarsh Dwivedi
  * [LinkedIn](https://www.linkedin.com/in/udwivedi/), [personal website](https://utkarshdwivedi.com/)
* Tested on: Windows 11 Home, AMD Ryzen 7 5800H @ 3.2GHz 16 GB, Nvidia GeForce RTX 3060 Laptop GPU 6000 MB

# Introduction

This project is an implementation of the scan and stream compaction algorithms, both sequentially on the CPU as well as in parallel on the GPU using CUDA, and a performance analysis and comparison of different implementations of the algorithm.

## Algorithms

### Scan
Scan computes the prefix sum of the array so that each element in the resulting array is the sum of all the elements occuring before it. This project contains the following implementations of scan:

1. **Sequential CPU scan**: A simple scan algorithm that runs sequentially on the CPU using a for loop.
2. **Naive parallel scan**: A scan algorithm that runs in parallel on the GPU and sums over adjacent elements while halving the array size each iteration
3. **Work efficient parallel scan**: A more performant parallel scan on the GPU that treats the array like a binary tree and runs in two steps: an "up-sweep" and then a "down-sweep"

### Stream Compaction
Stream compaction is an algorithm that, given an array and some condition, creates and returns a new array that contains only those elements from the original array that satisfy the given condition, and the order of the elements in the resulting array is preserved. In this project, stream compaction simply removes all `0s` from an array of integers, but it can easily be extended to work as a template on any given conditions. This project contains the following implementations of stream compaction:

1. **Simple CPU stream compaction**: A simple algorithm that runs sequentially on the CPU using a for loop and simply removes all elements that are 0
2. **CPU stream compaction using CPU scan**: A CPU side imitation of stream compaction in parallel, using the scan algorithm
3. **GPU stream compaction using GPU work efficient scan**: Proper implementation of stream compaction in parallel using the parallel scan algorithm on the GPU

# Performance Analysis

Include analysis, etc. (Remember, this is public, so don't put
anything here that you don't want to share with the world.)


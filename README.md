CUDA Stream Compaction
======================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 2**

* Shreyas Singh
  * [LinkedIn](https://linkedin.com/in/shreyassinghiitr), [Personal Website](https://github.com/shreyas3156).
* Tested on: Windows 10, i7-12700 @ 2.1GHz 32GB, T1000 (CETS Lab)

### About
This project features an implementation of all-prefix-sums operation on an array of data, often known as
_Scan_, followed by _Stream Compaction_, which refers to creating a new array with elements from the input that are filtered using a given criteria, preserving the order of elements in the process. We use _scan_ interchangeably with _exclusive scan_ throughout the project, where each element _k_ in the result array is the sum
of all elements up to but excluding _k_ itself in the input array.

We would see three different implementations of Scan, the first being a trivial sequential CPU-Scan and the other two being
"Naive" and "Work-Efficient" CUDA implementations of a parallel scan algorithm that leverage GPU's data parallelism, thus giving huge performance improvement
for large inputs.

Our Stream Compaction algorithm would remove `0`s from an array of `int`s, using CPU and Work-Efficient parallel scans using CUDA.
For a detailed version of Scan and further reading, check out [GPU Gems 3, Ch-39](https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda).


### Naive Parallel Scan

### Work-Efficient Parallel Scan

## Performance Analysis
Optimal Block size vs Different Scans
Number of elements vs Runtime
- POT runtime plots
- NPOT runtime
- 

### (TODO: Your README)

Include analysis, etc. (Remember, this is public, so don't put
anything here that you don't want to share with the world.)

### References


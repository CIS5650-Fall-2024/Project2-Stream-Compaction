CUDA Stream Compaction
======================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 2**

* Mengxuan Huang
  * [LinkedIn](https://www.linkedin.com/in/mengxuan-huang-52881624a/)
* Tested on: Windows 11, i9-13980HX @ 2.22GHz 64.0 GB, RTX4090-Laptop 16384MB

# Features
- CPU Scan
- CPU Compact
- GPU Scan Naive
- GPU Scan Efficient (Binary Balanced Tree)
- GPU Compact (use GPU effcient Scan)

# Analysis (Unoptimized)

| Block Size |
|-----------|
|   32      |

![](./img/runtime1.png)

It is noticed that to keep the scale of y-axis reasonable, I did not include $2^{24}$ data, which are over 10ms, of CPU method and GPU native in the graph.

According to the test result, CPU scan methods (looping) perform better than any GPU scan methods when the number of element is small. When the number of element increase, GPU methods require less time to finish scanning.

For the number of block, I only luanch neceeesary threads in both the naive scan method and the efficient scan method.
- In the naive scan method, only $n - 2^d$ threads needed for each iteration.
- In the up-sweep in the efficient scan method, $n / 2^{d+1}$ threads needed.
- In the down-sweep in the efficient scan method, $2^d$ threads needed.

## Block Size
Test on $2^{20}$ elements. Run 1000 times and compute the average run time.



## Compare

## Bottlenecks
The bottlenecks comes from the memory I/O.

## Output of a test
```
```
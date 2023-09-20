CUDA Stream Compaction
======================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 2**

Han Wang

Tested on: Windows 11, 11th Gen Intel(R) Core(TM) i9-11900H @ 2.50GHz 22GB, GTX 3070 Laptop GPU

### Analysis
**Roughly optimize the block sizes of each of your implementations for minimal run time on your GPU.**

**(You shouldn't compare unoptimized implementations to each other!)
Compare all of these GPU Scan implementations (Naive, Work-Efficient, and Thrust) to the serial CPU version of Scan. Plot a graph of the comparison (with array size on the independent axis).**

![Unlock FPS](img/graph1.png)

Based on my observation in my text, there are three phenomenons:
1. The block size seems to not influence the output that much.
2. The naive gpu approach is slower than the efficient approach.
3. Though I didn't plot out the output of the CPU scan, the CPU operation seems to be actually faster than the GPU operation.

The first 


**Don't mix up CpuTimer and GpuTimer.
To guess at what might be happening inside the Thrust implementation (e.g. allocation, memory copy), take a look at the Nsight timeline for its execution. Your analysis here doesn't have to be detailed, since you aren't even looking at the code for the implementation.
Write a brief explanation of the phenomena you see here.**



**
Can you find the performance bottlenecks? Is it memory I/O? Computation? Is it different for each implementation?
Paste the output of the test program into a triple-backtick block in your README.**

![Unlock FPS](img/output.png)

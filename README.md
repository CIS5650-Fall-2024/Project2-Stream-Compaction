**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 2**

* Alan Lee
  * [LinkedIn](https://www.linkedin.com/in/soohyun-alan-lee/)
* Tested on: Windows 10, AMD Ryzen 5 5600X 6-Core Processor @ 3.70GHz, 32GB RAM, NVIDIA GeForce RTX 3070 Ti (Personal Computer)

## CUDA Scan and Stream Compaction

### CPU Scan and Stream Compaction

### Naive CUDA Parallelization

### Work-Efficient CUDA Parallelization

### Thrust Library

### Radix Sort

## Running the Code

You should follow the regular setup guide as described in [Project 0](https://github.com/CIS5650-Fall-2024/Project0-Getting-Started/blob/main/INSTRUCTION.md#part-21-project-instructions---cuda).

`blockSize` for each implementation is defined at the top of respective `.cu` files.

Note that there may be illegal memory accesses for those implementations that compute index `k` if the array size and block sizes are too big such that there is an integer overflow. The limit is `blockSize 2` for `SIZE = 1 << 30`. Reducing `SIZE` by a factor of 2 (one less bit shift) allows `blockSize` to be doubled at most. (e.g. limit is `blockSize 8` for `SIZE = 1 << 28`).

## Performance Analysis

* Roughly optimize the block sizes of each of your implementations for minimal
  run time on your GPU.

The block size for each implementation was optimized to array size `1 << 20` (or 2^20). The relationships observed in the following analysis may be subject to change depending on the target of this optimization.

* Compare all of these GPU Scan implementations (Naive, Work-Efficient, and
  Thrust) to the serial CPU version of Scan. Plot a graph of the comparison
  (with array size on the independent axis).
  * We wrapped up both CPU and GPU timing functions as a performance timer class for you to conveniently measure the time cost.
    * We use `std::chrono` to provide CPU high-precision timing and CUDA event to measure the CUDA performance.
    * For CPU, put your CPU code between `timer().startCpuTimer()` and `timer().endCpuTimer()`.
    * For GPU, put your CUDA code between `timer().startGpuTimer()` and `timer().endGpuTimer()`. Be sure **not** to include any *initial/final* memory operations (`cudaMalloc`, `cudaMemcpy`) in your performance measurements, for comparability.
    * Don't mix up `CpuTimer` and `GpuTimer`.
  * To guess at what might be happening inside the Thrust implementation (e.g.
    allocation, memory copy), take a look at the Nsight timeline for its
    execution. Your analysis here doesn't have to be detailed, since you aren't
    even looking at the code for the implementation.



* Write a brief explanation of the phenomena you see here.
  * Can you find the performance bottlenecks? Is it memory I/O? Computation? Is
    it different for each implementation?

* Paste the output of the test program into a triple-backtick block in your
  README.
  * If you add your own tests (e.g. for radix sort or to test additional corner
    cases), be sure to mention it explicitly.

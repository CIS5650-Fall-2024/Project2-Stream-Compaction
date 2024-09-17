CUDA Stream Compaction
======================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 2 - Stream Compaction**

- Jordan Hochman
  - [LinkedIn](https://www.linkedin.com/in/jhochman24), [Personal Website](https://jordanh.xyz), [GitHub](https://github.com/JHawk0224)
- Tested on: Windows 11, Ryzen 7 5800 @ 3.4GHz 32GB, GeForce RTX 3060 Ti 8GB (Compute Capability: 8.6)

## Welcome to my Stream Compaction Project!

![nsight systems report timeline](img/entire-report.jpg)

This project investigates many different implementations of the scan and compact algorithms, along with optimizations. Additionally, it also implements [Radix sort](https://en.wikipedia.org/wiki/Radix_sort).

First I will explain what exactly scan and compact do, and then I'll walk through the different implementations of them. They both act on an array.

Scan computes the partial sums up to each element, and stores these partial sums either in another array or modifies the original in place. Additionally, there are two types of scans, inclusive and exclusive. Exclusive always starts with a 0, and the partial sums are only of all the elements strictly before the current index, while inclusive scan includes the current index. For example, for the list [4, 7, 12], the inclusive scan would be [4, 11, 23] while the exclusive scan would be [0, 4, 11].

Compact takes in an array and returns an array containing only the elements which meet a certain criteria, in the same order. For this project, that criteria will always be that it's non-zero. For example, Compacting [5, 2, 8, 3, 11, 9] where we only want the even elements would give the array [2, 8].

Now why do we care about these? Well at first, you might be tempted to implement each one as a simple list that iterates through the given array. While this works, and is serially the best we can do, we don't need to run them serially! By parallelizing these algorithms, we can make them run much faster. This is something that we want to do because these two algorithms occur very frequently in other areas.

This project implements scan and compact using many different methods, some on the CPU and some on the GPU. I will break down how each implementation works below, but feel free to check out `INSTRUCTION.md` for more details about scan and compact.

## Running the Implementations

If you are interested in running these implementations and the test cases for them, you may first want to ensure your computer is set up to do so. To do this, you can follow the instructions in [Project 0](https://github.com/JHawk0224/CIS5650-Project0-Getting-Started/blob/main/INSTRUCTION.md). After that, you may want to read through the [instructions](INSTRUCTION.md) for this project.

Additionally, you can tweak parameters found in `main.cpp` to change the program settings:

- `SIZE` - a power-of-2 size of array to test on
- `NPOT` - a non-power-of-2 size of array to test on
- `SORTMAXVAL` - the maximum possible value generated in the list for the sort test cases (only affects sorting!)

You may also be interested in changing `blockSize` in each of the following files: `naive.cu`, `efficient.cu`, `efficient-thread-optimized.cu`, and `radix-sort.cu`. This value affects the number of threads per block for each of these versions which are implemented on the GPU.

Note that before running the performance analysis, the block sizes were optimized for each implementation to give the best results, but feel free to change them. I explain more in depth how this was done in the corresponding section below. I will now walk through the different implementations of scan.

## Scan Implementations

### CPU Scan

In this approach, ...

### Naive Scan

In this approach, ...

### Work-Efficient Scan

### Thread Optimized Work-Efficient Scan

Mention part 5 EC
Show an example how it works, show how it is called and an example of its output

### Thrust Scan

### Memory Optimized Naive Scan

Mention part 6 EC 2
Show an example how it works, show how it is called and an example of its output

### Memory Optimized Work-Efficient Scan

Mention part 6 EC 2
Show an example how it works, show how it is called and an example of its output

Mention future thing to do avoid bank conflicts

## Compact Implementations

### CPU Compact With Scan

### CPU Compact Without Scan

### Compact with Work-Efficient Scan

### Compact with Thread Optimized Work-Efficient Scan

## Radix Sort

Mention part 6 EC 1
Show an example how it works, show how it is called and an example of its output

## Scan Performance Analysis

discuss how it was performed (average of 3)
discuss using timers from readme
discuss how memcpy and other operations weren't included

First I will discuss the measurement of performance used for the analysis below. They are all comparing FPS (frames per second), where the time is measured by the GLFW timer. The numbers given below correspond to the average FPS over the first 10 seconds of running the program (averaged over 3 tries due to the random generation of boid positions). To find the average FPS over the first 10 seconds, first the number of frames is stored (which is how many iterations of the loop finished), divided by the total time it took to compute all those frames (so not quite 10 but very close, as this should be the total time for every frame to have been computed).

For the below comparisons, they were all done in Release mode. The specs of the computer they were run on is mentioned at the very top. All of the raw data for these runs can be found in the `execution-data.xlsx` file [here](execution-data.xlsx), although screenshots are provided throughout for convenience.

Answer Q2 analysis part
Answer Q3

### Thrust Analysis

Answer Q2 thrust part

### Optimizing Block Sizes

Answer Q1

### Appendix

Here is the out from running the tests found in `main.cpp`. Note that I did add my own test cases for many of the extra methods I implemented. These include: ... mention exact tests.

INCLUDE IN PR extra tests implemented, ec features, changes to cmakelists

- description of project including list of features
- performance analysis
- extra credits documented, with performance comparison, show how it works (show how radix sort is called and output)

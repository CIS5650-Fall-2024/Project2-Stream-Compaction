CUDA Stream Compaction
======================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 2**

* Licheng CAO
  * [LinkedIn](https://www.linkedin.com/in/licheng-cao-6a523524b/)
* Tested on: Windows 10, i7-10870H @ 2.20GHz 32GB, GTX 3060 6009MB

Implemented Features
======================
  * naive GPU scan
  * efficient GPU scan (with reduced number of threads)
  * GPU stream compaction
  * naive GPU radix sort

Analysis
======================
### Blocksize selection
* ![blocksize_select](https://github.com/LichengCAO/Project2-Stream-Compaction/assets/81556019/7402b71e-c8dd-4949-9be1-cc63e8a7cec9)
* Figure 1
* Figure 1 shows the running time of my GPU program under different blocksizes. Consequently, I have chosen a block size of 128 for my naive method and 64 for my efficient method based on the results.

### Scan
* ![scan_largenum](https://github.com/LichengCAO/Project2-Stream-Compaction/assets/81556019/1dfd0dfe-80d0-440d-87a1-aab498bf6f9e)
 * Figure 2 average runtime with large array size
* ![scan_smallnum](https://github.com/LichengCAO/Project2-Stream-Compaction/assets/81556019/eca72f02-124d-4dbf-961e-b0e8864e4550)
 * Figure 3 average runtime with small array size
 * Figure 2 and 3 display the runtime performance of different methods for the scan operation. Upon analyzing these figures, it becomes evident that for array sizes below 24,576, the CPU method outperforms the other approaches in terms of speed. However, as the array size increases to approximately 100,000, the GPU methods exhibit superior performance. I think the bottlenecks in both GPU methods are related to memory input/output (I/O), as the computational tasks within these methods are not particularly complex. The bottleneck of CPU method may stem from its inability to execute operations in parallel, as its runtime is roughly proportional to the array size.
 * ![cuda_compute](https://github.com/LichengCAO/Project2-Stream-Compaction/assets/81556019/88dd91d1-cec9-45dc-873f-093b51e57935)
 * With Nsight Compute, we can see that thrust_scan uses 3 kernel functions to scan the array. I suspect that this method may closely resemble the scan method mentioned at the end of the slide, which involves dividing arrays into several blocks for scanning and subsequently adding offsets within each block to obtain the final result.

### Sort
* ![sort_largenum](https://github.com/LichengCAO/Project2-Stream-Compaction/assets/81556019/4e27cfed-501d-42f0-8029-8729402aff04)
 * Figure 4 average sort time with large array size
* ![sort_smallnum](https://github.com/LichengCAO/Project2-Stream-Compaction/assets/81556019/a9a22753-c58a-4664-9200-68ba61eb2641)
 * Figure 5 average sort time with small array size
* Figure 4, 5 show the runtime of sort with different method. With small arrays, my implementation of radix sort runs slower than the other two methods. With large large arrays, the 2 GPU methods run much faster than the CPU method.
* The primary computational cost in my implementation arises from the scan procedure used to rearrange numbers based on their bits. Initially, I employed two separate scans to determine the correct indices for numbers with '0' and '1' bits at a specific position. Surprisingly, this approach made my radix implementation even slower than the CPU method.
  * After reviewing others' implementations, I came to realize that I can calculate the index for numbers with '1' based on the scan result for numbers with '0' (i.e., index1 = total_number_of_0 + (cur_id_of_num - number_of_0_before_cur_id)). This modification boosted the performance of my implementation significantly, resulting in it running approximately 40% faster than the CPU method.
  * Furthermore, it became evident that scanning all 32 bits in each iteration was unnecessary for sorting numbers. By checking if the array is already sorted at the beginning of each loop, I could avoid unnecessary scans. As a result, the runtime for the sorting process reduced to just 1/8 of its original duration.


Tests
======================
```result
****************
** SCAN TESTS **
****************
==== cpu scan, power-of-two ====
   elapsed time: 96.4101ms    (std::chrono Measured)
==== cpu scan, non-power-of-two ====
   elapsed time: 92.3129ms    (std::chrono Measured)
    passed
==== naive scan, power-of-two ====
   elapsed time: 50.0737ms    (CUDA Measured)
    passed
==== naive scan, non-power-of-two ====
   elapsed time: 50.0838ms    (CUDA Measured)
    passed
==== work-efficient scan, power-of-two ====
   elapsed time: 18.0347ms    (CUDA Measured)
    passed
==== work-efficient scan, non-power-of-two ====
   elapsed time: 18.0009ms    (CUDA Measured)
    passed
==== thrust scan, power-of-two ====
   elapsed time: 1.9712ms    (CUDA Measured)
    passed
==== thrust scan, non-power-of-two ====
   elapsed time: 1.95203ms    (CUDA Measured)
    passed

*****************************
** STREAM COMPACTION TESTS **
*****************************
==== cpu compact without scan, power-of-two ====
   elapsed time: 138.717ms    (std::chrono Measured)
    passed
==== cpu compact without scan, non-power-of-two ====
   elapsed time: 137.777ms    (std::chrono Measured)
    passed
==== cpu compact with scan ====
   elapsed time: 233.015ms    (std::chrono Measured)
    passed
==== work-efficient compact, power-of-two ====
   elapsed time: 18.0009ms    (CUDA Measured)
    passed
==== work-efficient compact, non-power-of-two ====
   elapsed time: 18.0009ms    (CUDA Measured)
    passed

****************
** SORT TESTS **
****************
==== cpu sort, power-of-two ====
   elapsed time: 1192.6ms    (std::chrono Measured)
==== work-efficient sort, power-of-two ====
   elapsed time: 161.361ms    (CUDA Measured)
    passed
==== thrust sort, power-of-two ====
   elapsed time: 10.9355ms    (CUDA Measured)
    passed
==== cpu sort, non-power-of-two ====
   elapsed time: 1200.21ms    (std::chrono Measured)
==== work-efficient sort, non-power-of-two ====
   elapsed time: 159.406ms    (CUDA Measured)
    passed
==== thrust sort, non-power-of-two ====
   elapsed time: 10.8496ms    (CUDA Measured)
    passed
```


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
  * simple radix sort

Analysis
======================
### blocksize selection
* ![blocksize_select](https://github.com/LichengCAO/Project2-Stream-Compaction/assets/81556019/7402b71e-c8dd-4949-9be1-cc63e8a7cec9)
* Figure 1
* Figure 1 shows the running time of my GPU program under different blocksizes. As the result, I pick 128 blocksize for my naive method and 64 blocksize for my efficient method.

### scan
* ![scan_largenum](https://github.com/LichengCAO/Project2-Stream-Compaction/assets/81556019/1dfd0dfe-80d0-440d-87a1-aab498bf6f9e)
 * Figure 2 average runtime with large array size
* ![scan_smallnum](https://github.com/LichengCAO/Project2-Stream-Compaction/assets/81556019/eca72f02-124d-4dbf-961e-b0e8864e4550)
 * Figure 3 average runtime with small array size
 * Figure 2, 3 show the runtime of scan with different method. From figures, we can find that when the array size is under 24,576, the CPU method is faster than other methods. However, when the array size increase to around 100,000, the GPU methods perform better.



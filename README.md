CUDA Stream Compaction
======================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 2**

* Maya Diaz Huizar
* Tested on: Windows 10, R7-5800X @ 3.8GHz 32GB, RTX 3080 10GB 

### Questions
* Roughly optimize the block sizes of each of your implementations for minimal
  run time on your GPU.
  * Graphs:
  * The optimal block size for the CPU implementation is N/A.
  * ![image](<img/Naive GPU - Time (ms) vs Block Size (lower is better).png>)
  * ![image](<img/Efficient GPU - Time (ms) vs Block Size (lower is better).png>)

* Compare all of these GPU Scan implementations (Naive, Work-Efficient, and Thrust) to the serial CPU version of Scan. Plot a graph of the comparison (with array size on the independent axis).
  * ![image](<img/Various Scans - Time (ms) vs Element Count (lower is better).png>)
  * ![image](<img/Stream Compaction - Time (ms) vs Element Count (lower is better).png>)

  * Write a brief explanation of the phenomena you see here.
   * This generally makes sense, the efficient GPU scan and compact is much more efficient and more parallelizable, when compared to the naive approach. The CPU method is fast for small arrays and scales linearly, and thus is much worse at very large arrays when compared to the GPU implementation. Thrust almost certainly takes different approaches based on the size of the array, ensuring that it yields the best of both worlds, with fast small and large arrays. I also am wholly and entirely confident that the developers of the thrust library are more than capable of writing a faster library when compared to an undergrad CMPE major.
* Paste the output of the test program into a triple-backtick block in your README.
```
The below tests results are from scanning and steam compacting 2^29 element arrays.
****************
** SCAN TESTS **
****************
    [  42  34  32  22   8  16  34  39  37  30   7   2  14 ...   1   0 ]
==== cpu scan, power-of-two ====
   elapsed time: 246.098ms    (std::chrono Measured)
    [   0  42  76 108 130 138 154 188 227 264 294 301 303 ... 264144619 264144620 ]
==== cpu scan, non-power-of-two ====
   elapsed time: 245.254ms    (std::chrono Measured)
    [   0  42  76 108 130 138 154 188 227 264 294 301 303 ... 264144559 264144572 ]
    passed
==== naive scan, power-of-two ====
   elapsed time: 378.275ms    (CUDA Measured)
    passed
==== naive scan, non-power-of-two ====
   elapsed time: 377.967ms    (CUDA Measured)
    passed
==== work-efficient scan, power-of-two ====
   elapsed time: 17.0086ms    (CUDA Measured)
    passed
==== work-efficient scan, non-power-of-two ====
   elapsed time: 16.9234ms    (CUDA Measured)
    passed
==== thrust scan, power-of-two ====
   elapsed time: 7.19872ms    (CUDA Measured)
    passed
==== thrust scan, non-power-of-two ====
   elapsed time: 7.27962ms    (CUDA Measured)
    passed

*****************************
** STREAM COMPACTION TESTS **
*****************************
    [   3   0   2   1   3   2   0   0   3   3   0   3   2 ...   0   0 ]
==== cpu compact without scan, power-of-two ====
   elapsed time: 743.234ms    (std::chrono Measured)
    passed
==== cpu compact without scan, non-power-of-two ====
   elapsed time: 743.523ms    (std::chrono Measured)
    passed
==== cpu compact with scan ====
   elapsed time: 2002.52ms    (std::chrono Measured)
    passed
==== work-efficient compact, power-of-two ====
   elapsed time: 1130.99ms    (CUDA Measured)
    passed
==== work-efficient compact, non-power-of-two ====
   elapsed time: 875.497ms    (CUDA Measured)
    passed
```

* Extra Credit
  * My efficient GPU scan was efficient from the onset, but I also wasn't following the slides very closely. (5pt GPU approach)
  * I also implemented improvements for memory access to better align and thus prevent bank conflicts, based upon the overview provided by GPU Gems 3 Ch 39.2.3.

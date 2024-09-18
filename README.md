CUDA Stream Compaction
======================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 2**

* Akiko Zhu
  * [LinkedIn](https://www.linkedin.com/in/geming-akiko-zhu-b6705a255/)
* Tested on: Windows 11, i9-12900H @ 2.50GHz 16GB, RTX 3070Ti 8GB (Personal)

### Features
This project contains GPU parallel algorithms including Scan and Stream Compaction.
- A CPU-based Non-parallelized Exclusive Scan, Stream Compaction without Scan, and with Scan.
- A GPU-based Naive Parallel Scan.
- A GPU-based Work-Efficient Parallel Scan.
- A GPU-based Stream Compaction.
- Using Thrust library, implemented Exclusive Scan

### Results (Array size of 2^27)

```
****************
** SCAN TESTS **
****************
    [  11  33  25   7  42  33  18  40  18  45  40  44  10 ...   7   0 ]
==== cpu scan, power-of-two ====
   elapsed time: 198.072ms    (std::chrono Measured)
    [   0  11  44  69  76 118 151 169 209 227 272 312 356 ... -1007970099 -1007970092 ]
==== cpu scan, non-power-of-two ====
   elapsed time: 232.51ms    (std::chrono Measured)
    [   0  11  44  69  76 118 151 169 209 227 272 312 356 ... -1007970171 -1007970171 ]
    passed
==== naive scan, power-of-two ====
   elapsed time: 89.8661ms    (CUDA Measured)
    passed
==== naive scan, non-power-of-two ====
   elapsed time: 81.4132ms    (CUDA Measured)
    passed
==== work-efficient scan, power-of-two ====
   elapsed time: 60.4823ms    (CUDA Measured)
    passed
==== work-efficient scan, non-power-of-two ====
   elapsed time: 54.2374ms    (CUDA Measured)
    passed
==== thrust scan, power-of-two ====
   elapsed time: 97.2186ms    (CUDA Measured)
    passed
==== thrust scan, non-power-of-two ====
   elapsed time: 97.5937ms    (CUDA Measured)
    passed

*****************************
** STREAM COMPACTION TESTS **
*****************************
    [   2   3   0   0   2   2   2   3   0   1   1   2   1 ...   0   0 ]
==== cpu compact without scan, power-of-two ====
   elapsed time: 283.499ms    (std::chrono Measured)
    [   2   3   2   2   2   3   1   1   2   1   2   2   2 ...   2   1 ]
    passed
==== cpu compact without scan, non-power-of-two ====
   elapsed time: 270.86ms    (std::chrono Measured)
    [   2   3   2   2   2   3   1   1   2   1   2   2   2 ...   2   1 ]
    passed
==== cpu compact with scan ====
   elapsed time: 465.234ms    (std::chrono Measured)
    [   2   3   2   2   2   3   1   1   2   1   2   2   2 ...   2   1 ]
    passed
==== work-efficient compact, power-of-two ====
   elapsed time: 72.7748ms    (CUDA Measured)
    passed
==== work-efficient compact, non-power-of-two ====
   elapsed time: 64.1773ms    (CUDA Measured)
    passed
```
    

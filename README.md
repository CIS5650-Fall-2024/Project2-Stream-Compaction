CUDA Stream Compaction
======================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 2**

* (TODO) YOUR NAME HERE
  * (TODO) [LinkedIn](), [personal website](), [twitter](), etc.
* Tested on: (TODO) Windows 22, i7-2222 @ 2.22GHz 22GB, GTX 222 222MB (Moore 2222 Lab)

### (TODO: Your README)

CPU only:
9/16/23 results:

****************
** SCAN TESTS **
****************
    [  24   9  33  25  21   8  37  44  20   0   3   0  39 ...  27   0 ]
==== cpu scan, power-of-two ====
   elapsed time: 0.0004ms    (std::chrono Measured)
    [   0  24  33  66  91 112 120 157 201 221 221 224 224 ... 5878 5905 ]
==== cpu scan, non-power-of-two ====
   elapsed time: 0.0004ms    (std::chrono Measured)
    [   0  24  33  66  91 112 120 157 201 221 221 224 224 ... 5832 5856 ]
    passed
==== naive scan, power-of-two ====
   elapsed time: 0.007168ms    (CUDA Measured)
    a[1] = 24, b[1] = 0
    FAIL VALUE
==== naive scan, non-power-of-two ====
   elapsed time: 0.002048ms    (CUDA Measured)
    a[1] = 24, b[1] = 0
    FAIL VALUE
==== work-efficient scan, power-of-two ====
   elapsed time: 0.003072ms    (CUDA Measured)
    a[1] = 24, b[1] = 0
    FAIL VALUE
==== work-efficient scan, non-power-of-two ====
   elapsed time: 0.002048ms    (CUDA Measured)
    a[1] = 24, b[1] = 0
    FAIL VALUE
==== thrust scan, power-of-two ====
   elapsed time: 0.003072ms    (CUDA Measured)
    a[1] = 24, b[1] = 0
    FAIL VALUE
==== thrust scan, non-power-of-two ====
   elapsed time: 0.003072ms    (CUDA Measured)
    a[1] = 24, b[1] = 0
    FAIL VALUE

*****************************
** STREAM COMPACTION TESTS **
*****************************
    [   0   3   1   1   1   0   1   0   0   2   1   2   1 ...   1   0 ]
==== cpu compact without scan, power-of-two ====
   elapsed time: 0.0009ms    (std::chrono Measured)
    [   3   1   1   1   1   2   1   2   1   3   2   3   3 ...   3   1 ]
    passed
==== cpu compact without scan, non-power-of-two ====
   elapsed time: 0.0004ms    (std::chrono Measured)
    [   3   1   1   1   1   2   1   2   1   3   2   3   3 ...   2   3 ]
    passed
==== cpu compact with scan ====
   elapsed time: 0.001ms    (std::chrono Measured)
    [   3   1   1   1   1   2   1   2   1   3   2   3   3 ...   3   1 ]
    passed
==== work-efficient compact, power-of-two ====
   elapsed time: 0.015328ms    (CUDA Measured)
    expected 184 elements, got -1
    FAIL COUNT
==== work-efficient compact, non-power-of-two ====
   elapsed time: 0.003072ms    (CUDA Measured)
    expected 182 elements, got -1
    FAIL COUNT


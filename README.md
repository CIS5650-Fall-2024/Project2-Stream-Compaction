CUDA Stream Compaction
======================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 2**

* (TODO) YOUR NAME HERE
  * (TODO) [LinkedIn](), [personal website](), [twitter](), etc.
* Tested on: (TODO) Windows 22, i7-2222 @ 2.22GHz 22GB, GTX 222 222MB (Moore 2222 Lab)

### (TODO: Your README)


9/16/23 test results: 

1. CPU only: arr size = 1 << 8

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


2. arr size = 1 << 28

****************
** SCAN TESTS **
****************
    [  38   3  45  32  46  48  42   7  15  10  10  16  45 ...  23   0 ]
==== cpu scan, power-of-two ====
   elapsed time: 121.7ms    (std::chrono Measured)
    [   0  38  41  86 118 164 212 254 261 276 286 296 312 ... -2015407160 -2015407137 ]
==== cpu scan, non-power-of-two ====
   elapsed time: 126.34ms    (std::chrono Measured)
    [   0  38  41  86 118 164 212 254 261 276 286 296 312 ... -2015407204 -2015407190 ]
    passed

*****************************
** STREAM COMPACTION TESTS **
*****************************
    [   1   1   2   3   0   3   0   2   0   2   2   0   3 ...   0   0 ]
==== cpu compact without scan, power-of-two ====
   elapsed time: 610.69ms    (std::chrono Measured)
    [   1   1   2   3   3   2   2   2   3   3   2   1   2 ...   1   3 ]
    passed
==== cpu compact without scan, non-power-of-two ====
   elapsed time: 636.085ms    (std::chrono Measured)
    [   1   1   2   3   3   2   2   2   3   3   2   1   2 ...   3   1 ]
    passed
==== cpu compact with scan ====
   elapsed time: 1088.41ms    (std::chrono Measured)
    [   1   1   2   3   3   2   2   2   3   3   2   1   2 ...   1   3 ]
    passed


3. gpu naive scan: arr size = 1 << 8

****************
** SCAN TESTS **
****************
    [  47  28  21   1  30  20   6  37   4  24   4  11  26 ...  11   0 ]
==== cpu scan, power-of-two ====
   elapsed time: 0.0005ms    (std::chrono Measured)
    [   0  47  75  96  97 127 147 153 190 194 218 222 233 ... 6240 6251 ]
==== cpu scan, non-power-of-two ====
   elapsed time: 0.0003ms    (std::chrono Measured)
    [   0  47  75  96  97 127 147 153 190 194 218 222 233 ... 6127 6151 ]
    passed
==== naive scan, power-of-two ====
   elapsed time: 0.1024ms    (CUDA Measured)
    passed
==== naive scan, non-power-of-two ====
   elapsed time: 0.034816ms    (CUDA Measured)
    passed

*****************************
** STREAM COMPACTION TESTS **
*****************************
    [   1   0   3   1   0   0   2   3   0   2   0   1   0 ...   1   0 ]
==== cpu compact without scan, power-of-two ====
   elapsed time: 0.0009ms    (std::chrono Measured)
    [   1   3   1   2   3   2   1   3   3   3   3   1   2 ...   1   1 ]
    passed
==== cpu compact without scan, non-power-of-two ====
   elapsed time: 0.0007ms    (std::chrono Measured)
    [   1   3   1   2   3   2   1   3   3   3   3   1   2 ...   2   2 ]
    passed
==== cpu compact with scan ====
   elapsed time: 0.0008ms    (std::chrono Measured)
    [   1   3   1   2   3   2   1   3   3   3   3   1   2 ...   1   1 ]
    passed

4. arr size = 1 << 28:

****************
** SCAN TESTS **
****************
    [  43  29  30   8  31  19  46  25  29  39  44  30   1 ...   9   0 ]
==== cpu scan, power-of-two ====
   elapsed time: 210.313ms    (std::chrono Measured)
    [   0  43  72 102 110 141 160 206 231 260 299 343 373 ... -2015277306 -2015277297 ]
==== cpu scan, non-power-of-two ====
   elapsed time: 122.004ms    (std::chrono Measured)
    [   0  43  72 102 110 141 160 206 231 260 299 343 373 ... -2015277354 -2015277330 ]
    passed
==== naive scan, power-of-two ====
   elapsed time: 195.855ms    (CUDA Measured)
    passed
==== naive scan, non-power-of-two ====
   elapsed time: 195.335ms    (CUDA Measured)
    passed

*****************************
** STREAM COMPACTION TESTS **
*****************************
    [   3   1   2   1   2   3   0   3   3   3   2   2   3 ...   3   0 ]
==== cpu compact without scan, power-of-two ====
   elapsed time: 611.903ms    (std::chrono Measured)
    [   3   1   2   1   2   3   3   3   3   2   2   3   1 ...   3   3 ]
    passed
==== cpu compact without scan, non-power-of-two ====
   elapsed time: 609.371ms    (std::chrono Measured)
    [   3   1   2   1   2   3   3   3   3   2   2   3   1 ...   2   3 ]
    passed
==== cpu compact with scan ====
   elapsed time: 1105.18ms    (std::chrono Measured)
    [   3   1   2   1   2   3   3   3   3   2   2   3   1 ...   3   3 ]
    passed
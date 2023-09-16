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

5. non-power-of-2: arr size = 1 << 8 + 1

****************
** SCAN TESTS **
****************
    [  41  33   3   5  39  13  38   3  21   7  22  40  18 ...   5   0 ]
==== cpu scan, power-of-two ====
   elapsed time: 0.0004ms    (std::chrono Measured)
    [   0  41  74  77  82 121 134 172 175 196 203 225 265 ... 12372 12377 ]
==== cpu scan, non-power-of-two ====
   elapsed time: 0.0004ms    (std::chrono Measured)
    [   0  41  74  77  82 121 134 172 175 196 203 225 265 ... 12294 12324 ]
    passed
==== naive scan, power-of-two ====
   elapsed time: 0.04096ms    (CUDA Measured)
    passed
==== naive scan, non-power-of-two ====
   elapsed time: 0.06144ms    (CUDA Measured)
    passed

*****************************
** STREAM COMPACTION TESTS **
*****************************
    [   3   3   3   3   1   3   0   3   3   3   2   0   0 ...   3   0 ]
==== cpu compact without scan, power-of-two ====
   elapsed time: 0.0016ms    (std::chrono Measured)
    [   3   3   3   3   1   3   3   3   3   2   1   1   3 ...   2   3 ]
    passed
==== cpu compact without scan, non-power-of-two ====
   elapsed time: 0.0019ms    (std::chrono Measured)
    [   3   3   3   3   1   3   3   3   3   2   1   1   3 ...   2   2 ]
    passed
==== cpu compact with scan ====
   elapsed time: 0.002ms    (std::chrono Measured)
    [   3   3   3   3   1   3   3   3   3   2   1   1   3 ...   2   3 ]
    passed

6. non-power-of-2: arr size = 10,000

****************
** SCAN TESTS **
****************
    [  22  38  29  17  26  43  48  36  14  45  13  30  28 ...   8   0 ]
==== cpu scan, power-of-two ====
   elapsed time: 0.0064ms    (std::chrono Measured)
    [   0  22  60  89 106 132 175 223 259 273 318 331 361 ... 244061 244069 ]
==== cpu scan, non-power-of-two ====
   elapsed time: 0.0043ms    (std::chrono Measured)
    [   0  22  60  89 106 132 175 223 259 273 318 331 361 ... 243956 243991 ]
    passed
==== naive scan, power-of-two ====
   elapsed time: 0.162816ms    (CUDA Measured)
    passed
==== naive scan, non-power-of-two ====
   elapsed time: 0.161792ms    (CUDA Measured)
    passed

*****************************
** STREAM COMPACTION TESTS **
*****************************
    [   0   0   1   1   0   1   0   0   2   3   3   0   0 ...   0   0 ]
==== cpu compact without scan, power-of-two ====
   elapsed time: 0.0254ms    (std::chrono Measured)
    [   1   1   1   2   3   3   2   2   1   1   3   2   2 ...   1   2 ]
    passed
==== cpu compact without scan, non-power-of-two ====
   elapsed time: 0.0278ms    (std::chrono Measured)
    [   1   1   1   2   3   3   2   2   1   1   3   2   2 ...   1   2 ]
    passed
==== cpu compact with scan ====
   elapsed time: 0.0428ms    (std::chrono Measured)
    [   1   1   1   2   3   3   2   2   1   1   3   2   2 ...   1   2 ]
    passed

7. non-power-of-2: arr size = 1,000,000

****************
** SCAN TESTS **
****************
    [  11  46  46  35  44  22  32  39  13  39  12  22  14 ...   5   0 ]
==== cpu scan, power-of-two ====
   elapsed time: 0.4483ms    (std::chrono Measured)
    [   0  11  57 103 138 182 204 236 275 288 327 339 361 ... 24484977 24484982 ]
==== cpu scan, non-power-of-two ====
   elapsed time: 0.4603ms    (std::chrono Measured)
    [   0  11  57 103 138 182 204 236 275 288 327 339 361 ... 24484907 24484925 ]
    passed
==== naive scan, power-of-two ====
   elapsed time: 0.56752ms    (CUDA Measured)
    passed
==== naive scan, non-power-of-two ====
   elapsed time: 1.10698ms    (CUDA Measured)
    passed

*****************************
** STREAM COMPACTION TESTS **
*****************************
    [   1   0   2   3   2   2   2   3   1   3   2   2   2 ...   1   0 ]
==== cpu compact without scan, power-of-two ====
   elapsed time: 2.5808ms    (std::chrono Measured)
    [   1   2   3   2   2   2   3   1   3   2   2   2   3 ...   2   1 ]
    passed
==== cpu compact without scan, non-power-of-two ====
   elapsed time: 2.3386ms    (std::chrono Measured)
    [   1   2   3   2   2   2   3   1   3   2   2   2   3 ...   3   2 ]
    passed
==== cpu compact with scan ====
   elapsed time: 3.7589ms    (std::chrono Measured)
    [   1   2   3   2   2   2   3   1   3   2   2   2   3 ...   2   1 ]
    passed


8. GPU work-efficient scan: arr size = 1 << 8

****************
** SCAN TESTS **
****************
    [  44  45  47   1  26  14  41  13  30  19  12  45  18 ...   7   0 ]
==== cpu scan, power-of-two ====
   elapsed time: 0.0004ms    (std::chrono Measured)
    [   0  44  89 136 137 163 177 218 231 261 280 292 337 ... 6600 6607 ]
==== cpu scan, non-power-of-two ====
   elapsed time: 0.0002ms    (std::chrono Measured)
    [   0  44  89 136 137 163 177 218 231 261 280 292 337 ... 6528 6561 ]
    passed
==== naive scan, power-of-two ====
   elapsed time: 0.054272ms    (CUDA Measured)
    passed
==== naive scan, non-power-of-two ====
   elapsed time: 0.034816ms    (CUDA Measured)
    passed
==== work-efficient scan, power-of-two ====
   elapsed time: 5.25005ms    (CUDA Measured)
    passed
==== work-efficient scan, non-power-of-two ====
   elapsed time: 0.325696ms    (CUDA Measured)
    passed

*****************************
** STREAM COMPACTION TESTS **
*****************************
    [   2   3   3   3   0   0   3   3   2   1   2   1   0 ...   1   0 ]
==== cpu compact without scan, power-of-two ====
   elapsed time: 0.0074ms    (std::chrono Measured)
    [   2   3   3   3   3   3   2   1   2   1   1   1   2 ...   3   1 ]
    passed
==== cpu compact without scan, non-power-of-two ====
   elapsed time: 0.0007ms    (std::chrono Measured)
    [   2   3   3   3   3   3   2   1   2   1   1   1   2 ...   3   3 ]
    passed
==== cpu compact with scan ====
   elapsed time: 0.0009ms    (std::chrono Measured)
    [   2   3   3   3   3   3   2   1   2   1   1   1   2 ...   3   1 ]
    passed

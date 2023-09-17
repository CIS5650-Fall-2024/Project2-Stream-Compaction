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

9. arr size = 1 << 28

****************
** SCAN TESTS **
****************
    [  39  17  23   3  26  20  33  24  41  30  41  31  21 ...   1   0 ]
==== cpu scan, power-of-two ====
   elapsed time: 129.741ms    (std::chrono Measured)
    [   0  39  56  79  82 108 128 161 185 226 256 297 328 ... -2015586837 -2015586836 ]
==== cpu scan, non-power-of-two ====
   elapsed time: 127.985ms    (std::chrono Measured)
    [   0  39  56  79  82 108 128 161 185 226 256 297 328 ... -2015586885 -2015586869 ]
    passed
==== naive scan, power-of-two ====
   elapsed time: 196.371ms    (CUDA Measured)
    passed
==== naive scan, non-power-of-two ====
   elapsed time: 194.958ms    (CUDA Measured)
    passed
==== work-efficient scan, power-of-two ====
   elapsed time: 140.183ms    (CUDA Measured)
    passed
==== work-efficient scan, non-power-of-two ====
   elapsed time: 141.531ms    (CUDA Measured)
    passed

*****************************
** STREAM COMPACTION TESTS **
*****************************
    [   2   0   1   0   0   3   1   1   2   2   2   2   3 ...   2   0 ]
==== cpu compact without scan, power-of-two ====
   elapsed time: 631.703ms    (std::chrono Measured)
    [   2   1   3   1   1   2   2   2   2   3   2   2   1 ...   2   2 ]
    passed
==== cpu compact without scan, non-power-of-two ====
   elapsed time: 612.56ms    (std::chrono Measured)
    [   2   1   3   1   1   2   2   2   2   3   2   2   1 ...   1   2 ]
    passed
==== cpu compact with scan ====
   elapsed time: 978.363ms    (std::chrono Measured)
    [   2   1   3   1   1   2   2   2   2   3   2   2   1 ...   2   2 ]
    passed


10. arr size = 10,000

****************
** SCAN TESTS **
****************
    [  35  33  18  22  28   2  45   5  29  47  47  39  24 ...  31   0 ]
==== cpu scan, power-of-two ====
   elapsed time: 0.005ms    (std::chrono Measured)
    [   0  35  68  86 108 136 138 183 188 217 264 311 350 ... 244314 244345 ]
==== cpu scan, non-power-of-two ====
   elapsed time: 0.0051ms    (std::chrono Measured)
    [   0  35  68  86 108 136 138 183 188 217 264 311 350 ... 244242 244254 ]
    passed
==== naive scan, power-of-two ====
   elapsed time: 0.123904ms    (CUDA Measured)
    passed
==== naive scan, non-power-of-two ====
   elapsed time: 0.159744ms    (CUDA Measured)
    passed
==== work-efficient scan, power-of-two ====
   elapsed time: 0.43008ms    (CUDA Measured)
    passed
==== work-efficient scan, non-power-of-two ====
   elapsed time: 0.356352ms    (CUDA Measured)
    passed

*****************************
** STREAM COMPACTION TESTS **
*****************************
    [   3   1   2   0   2   2   1   3   3   1   1   3   0 ...   1   0 ]
==== cpu compact without scan, power-of-two ====
   elapsed time: 0.0269ms    (std::chrono Measured)
    [   3   1   2   2   2   1   3   3   1   1   3   1   2 ...   1   1 ]
    passed
==== cpu compact without scan, non-power-of-two ====
   elapsed time: 0.0237ms    (std::chrono Measured)
    [   3   1   2   2   2   1   3   3   1   1   3   1   2 ...   2   3 ]
    passed
==== cpu compact with scan ====
   elapsed time: 0.0371ms    (std::chrono Measured)
    [   3   1   2   2   2   1   3   3   1   1   3   1   2 ...   1   1 ]
    passed

11. arr size = 1,000,000

****************
** SCAN TESTS **
****************
    [  36  32  24  33  40  22  28  16   8  30  44  12  42 ...   7   0 ]
==== cpu scan, power-of-two ====
   elapsed time: 0.4109ms    (std::chrono Measured)
    [   0  36  68  92 125 165 187 215 231 239 269 313 325 ... 24478746 24478753 ]
==== cpu scan, non-power-of-two ====
   elapsed time: 0.3959ms    (std::chrono Measured)
    [   0  36  68  92 125 165 187 215 231 239 269 313 325 ... 24478683 24478692 ]
    passed
==== naive scan, power-of-two ====
   elapsed time: 0.574304ms    (CUDA Measured)
    passed
==== naive scan, non-power-of-two ====
   elapsed time: 0.59184ms    (CUDA Measured)
    passed
==== work-efficient scan, power-of-two ====
   elapsed time: 1.01581ms    (CUDA Measured)
    passed
==== work-efficient scan, non-power-of-two ====
   elapsed time: 0.689152ms    (CUDA Measured)
    passed

*****************************
** STREAM COMPACTION TESTS **
*****************************
    [   0   2   2   1   2   0   2   0   0   0   0   0   2 ...   3   0 ]
==== cpu compact without scan, power-of-two ====
   elapsed time: 2.8841ms    (std::chrono Measured)
    [   2   2   1   2   2   2   2   3   1   3   2   1   3 ...   1   3 ]
    passed
==== cpu compact without scan, non-power-of-two ====
   elapsed time: 2.5874ms    (std::chrono Measured)
    [   2   2   1   2   2   2   2   3   1   3   2   1   3 ...   1   1 ]
    passed
==== cpu compact with scan ====
   elapsed time: 3.7957ms    (std::chrono Measured)
    [   2   2   1   2   2   2   2   3   1   3   2   1   3 ...   1   3 ]
    passed


12. GPU work-efficient stream compaction: arr size = 1 << 8

****************
** SCAN TESTS **
****************
    [  44  38  20  40  28   0   6  24  16  28  14  18  15 ...  27   0 ]
==== cpu scan, power-of-two ====
   elapsed time: 0.0004ms    (std::chrono Measured)
    [   0  44  82 102 142 170 170 176 200 216 244 258 276 ... 6497 6524 ]
==== cpu scan, non-power-of-two ====
   elapsed time: 0.0003ms    (std::chrono Measured)
    [   0  44  82 102 142 170 170 176 200 216 244 258 276 ... 6443 6448 ]
    passed
==== naive scan, power-of-two ====
   elapsed time: 0.03584ms    (CUDA Measured)
    passed
==== naive scan, non-power-of-two ====
   elapsed time: 0.06144ms    (CUDA Measured)
    passed
==== work-efficient scan, power-of-two ====
   elapsed time: 0.139264ms    (CUDA Measured)
    passed
==== work-efficient scan, non-power-of-two ====
   elapsed time: 0.165888ms    (CUDA Measured)
    passed
==== thrust scan, power-of-two ====
   elapsed time: 0.002944ms    (CUDA Measured)
    a[1] = 44, b[1] = 0
    FAIL VALUE
==== thrust scan, non-power-of-two ====
   elapsed time: 0.004352ms    (CUDA Measured)
    a[1] = 44, b[1] = 0
    FAIL VALUE

*****************************
** STREAM COMPACTION TESTS **
*****************************
    [   0   2   0   2   2   0   2   0   0   0   2   0   1 ...   3   0 ]
==== cpu compact without scan, power-of-two ====
   elapsed time: 0.0088ms    (std::chrono Measured)
    [   2   2   2   2   2   1   2   3   1   3   2   1   3 ...   1   3 ]
    passed
==== cpu compact without scan, non-power-of-two ====
   elapsed time: 0.0006ms    (std::chrono Measured)
    [   2   2   2   2   2   1   2   3   1   3   2   1   3 ...   3   3 ]
    passed
==== cpu compact with scan ====
   elapsed time: 0.0013ms    (std::chrono Measured)
    [   2   2   2   2   2   1   2   3   1   3   2   1   3 ...   1   3 ]
    passed
==== work-efficient compact, power-of-two ====
   elapsed time: 0.265216ms    (CUDA Measured)
    passed
==== work-efficient compact, non-power-of-two ====
   elapsed time: 0.18432ms    (CUDA Measured)
    passed

13. arr size = 1 << 28

****************
** SCAN TESTS **
****************
    [  31   8  39   3  25  18  30   9   8  38  32  33  35 ...  40   0 ]
==== cpu scan, power-of-two ====
   elapsed time: 115.683ms    (std::chrono Measured)
    [   0  31  39  78  81 106 124 154 163 171 209 241 274 ... -2016042912 -2016042872 ]
==== cpu scan, non-power-of-two ====
   elapsed time: 117.459ms    (std::chrono Measured)
    [   0  31  39  78  81 106 124 154 163 171 209 241 274 ... -2016042942 -2016042934 ]
    passed
==== naive scan, power-of-two ====
   elapsed time: 196.181ms    (CUDA Measured)
    passed
==== naive scan, non-power-of-two ====
   elapsed time: 195.97ms    (CUDA Measured)
    passed
==== work-efficient scan, power-of-two ====
   elapsed time: 140.868ms    (CUDA Measured)
    passed
==== work-efficient scan, non-power-of-two ====
   elapsed time: 140.192ms    (CUDA Measured)
    passed
==== thrust scan, power-of-two ====
   elapsed time: 0.00272ms    (CUDA Measured)
    a[1] = 31, b[1] = 0
    FAIL VALUE
==== thrust scan, non-power-of-two ====
   elapsed time: 0.003296ms    (CUDA Measured)
    a[1] = 31, b[1] = 0
    FAIL VALUE

*****************************
** STREAM COMPACTION TESTS **
*****************************
    [   1   0   0   1   3   1   1   2   3   0   0   2   0 ...   1   0 ]
==== cpu compact without scan, power-of-two ====
   elapsed time: 611.492ms    (std::chrono Measured)
    [   1   1   3   1   1   2   3   2   2   3   2   2   1 ...   3   1 ]
    passed
==== cpu compact without scan, non-power-of-two ====
   elapsed time: 605.904ms    (std::chrono Measured)
    [   1   1   3   1   1   2   3   2   2   3   2   2   1 ...   2   1 ]
    passed
==== cpu compact with scan ====
   elapsed time: 965.131ms    (std::chrono Measured)
    [   1   1   3   1   1   2   3   2   2   3   2   2   1 ...   3   1 ]
    passed
==== work-efficient compact, power-of-two ====
   elapsed time: 168.418ms    (CUDA Measured)
    passed
==== work-efficient compact, non-power-of-two ====
   elapsed time: 163.62ms    (CUDA Measured)
    passed

14. arr size = 10,000

****************
** SCAN TESTS **
****************
    [  11   9   5  19  37  18  43  45  48  43  31  21  13 ...  43   0 ]
==== cpu scan, power-of-two ====
   elapsed time: 0.0055ms    (std::chrono Measured)
    [   0  11  20  25  44  81  99 142 187 235 278 309 330 ... 245583 245626 ]
==== cpu scan, non-power-of-two ====
   elapsed time: 0.0035ms    (std::chrono Measured)
    [   0  11  20  25  44  81  99 142 187 235 278 309 330 ... 245509 245540 ]
    passed
==== naive scan, power-of-two ====
   elapsed time: 0.121856ms    (CUDA Measured)
    passed
==== naive scan, non-power-of-two ====
   elapsed time: 0.082944ms    (CUDA Measured)
    passed
==== work-efficient scan, power-of-two ====
   elapsed time: 0.306176ms    (CUDA Measured)
    passed
==== work-efficient scan, non-power-of-two ====
   elapsed time: 0.283648ms    (CUDA Measured)
    passed
==== thrust scan, power-of-two ====
   elapsed time: 0.002912ms    (CUDA Measured)
    a[1] = 11, b[1] = 0
    FAIL VALUE
==== thrust scan, non-power-of-two ====
   elapsed time: 0.00336ms    (CUDA Measured)
    a[1] = 11, b[1] = 0
    FAIL VALUE

*****************************
** STREAM COMPACTION TESTS **
*****************************
    [   2   2   3   2   0   0   1   2   1   3   0   1   0 ...   2   0 ]
==== cpu compact without scan, power-of-two ====
   elapsed time: 0.0257ms    (std::chrono Measured)
    [   2   2   3   2   1   2   1   3   1   1   3   3   3 ...   3   2 ]
    passed
==== cpu compact without scan, non-power-of-two ====
   elapsed time: 0.2444ms    (std::chrono Measured)
    [   2   2   3   2   1   2   1   3   1   1   3   3   3 ...   2   3 ]
    passed
==== cpu compact with scan ====
   elapsed time: 0.0399ms    (std::chrono Measured)
    [   2   2   3   2   1   2   1   3   1   1   3   3   3 ...   3   2 ]
    passed
==== work-efficient compact, power-of-two ====
   elapsed time: 0.258048ms    (CUDA Measured)
    passed
==== work-efficient compact, non-power-of-two ====
   elapsed time: 0.31232ms    (CUDA Measured)
    passed

15. arr size = 1,000,000

****************
** SCAN TESTS **
****************
    [   4  19  46  28  26  36  18  17  44  22  24   9  34 ...   5   0 ]
==== cpu scan, power-of-two ====
   elapsed time: 0.4287ms    (std::chrono Measured)
    [   0   4  23  69  97 123 159 177 194 238 260 284 293 ... 24491262 24491267 ]
==== cpu scan, non-power-of-two ====
   elapsed time: 0.5807ms    (std::chrono Measured)
    [   0   4  23  69  97 123 159 177 194 238 260 284 293 ... 24491225 24491242 ]
    passed
==== naive scan, power-of-two ====
   elapsed time: 0.566336ms    (CUDA Measured)
    passed
==== naive scan, non-power-of-two ====
   elapsed time: 0.72272ms    (CUDA Measured)
    passed
==== work-efficient scan, power-of-two ====
   elapsed time: 0.731136ms    (CUDA Measured)
    passed
==== work-efficient scan, non-power-of-two ====
   elapsed time: 0.833536ms    (CUDA Measured)
    passed
==== thrust scan, power-of-two ====
   elapsed time: 0.004448ms    (CUDA Measured)
    a[1] = 4, b[1] = 0
    FAIL VALUE
==== thrust scan, non-power-of-two ====
   elapsed time: 0.005696ms    (CUDA Measured)
    a[1] = 4, b[1] = 0
    FAIL VALUE

*****************************
** STREAM COMPACTION TESTS **
*****************************
    [   2   3   2   0   2   2   0   1   0   0   0   3   0 ...   3   0 ]
==== cpu compact without scan, power-of-two ====
   elapsed time: 2.6299ms    (std::chrono Measured)
    [   2   3   2   2   2   1   3   1   2   1   2   3   1 ...   2   3 ]
    passed
==== cpu compact without scan, non-power-of-two ====
   elapsed time: 2.6916ms    (std::chrono Measured)
    [   2   3   2   2   2   1   3   1   2   1   2   3   1 ...   3   2 ]
    passed
==== cpu compact with scan ====
   elapsed time: 3.7679ms    (std::chrono Measured)
    [   2   3   2   2   2   1   3   1   2   1   2   3   1 ...   2   3 ]
    passed
==== work-efficient compact, power-of-two ====
   elapsed time: 0.99248ms    (CUDA Measured)
    passed
==== work-efficient compact, non-power-of-two ====
   elapsed time: 0.84928ms    (CUDA Measured)
    passed


16. Thrust scan: arr size = 1 << 8

****************
** SCAN TESTS **
****************
    [  40  24  11  48  47  39  48  35  20  17  21  11  16 ...  31   0 ]
==== cpu scan, power-of-two ====
   elapsed time: 0.0005ms    (std::chrono Measured)
    [   0  40  64  75 123 170 209 257 292 312 329 350 361 ... 6620 6651 ]
==== cpu scan, non-power-of-two ====
   elapsed time: 0.0003ms    (std::chrono Measured)
    [   0  40  64  75 123 170 209 257 292 312 329 350 361 ... 6566 6610 ]
    passed
==== naive scan, power-of-two ====
   elapsed time: 0.055296ms    (CUDA Measured)
    passed
==== naive scan, non-power-of-two ====
   elapsed time: 0.054272ms    (CUDA Measured)
    passed
==== work-efficient scan, power-of-two ====
   elapsed time: 0.14848ms    (CUDA Measured)
    passed
==== work-efficient scan, non-power-of-two ====
   elapsed time: 0.157696ms    (CUDA Measured)
    passed
==== thrust scan, power-of-two ====
   elapsed time: 0.472064ms    (CUDA Measured)
    passed
==== thrust scan, non-power-of-two ====
   elapsed time: 0.031744ms    (CUDA Measured)
    passed

*****************************
** STREAM COMPACTION TESTS **
*****************************
    [   0   0   1   2   1   3   0   1   0   1   3   3   0 ...   3   0 ]
==== cpu compact without scan, power-of-two ====
   elapsed time: 0.0007ms    (std::chrono Measured)
    [   1   2   1   3   1   1   3   3   3   2   3   3   3 ...   3   3 ]
    passed
==== cpu compact without scan, non-power-of-two ====
   elapsed time: 0.0006ms    (std::chrono Measured)
    [   1   2   1   3   1   1   3   3   3   2   3   3   3 ...   3   3 ]
    passed
==== cpu compact with scan ====
   elapsed time: 0.0011ms    (std::chrono Measured)
    [   1   2   1   3   1   1   3   3   3   2   3   3   3 ...   3   3 ]
    passed
==== work-efficient compact, power-of-two ====
   elapsed time: 0.188416ms    (CUDA Measured)
    passed
==== work-efficient compact, non-power-of-two ====
   elapsed time: 0.185344ms    (CUDA Measured)
    passed


17. arr size = 1 << 28

****************
** SCAN TESTS **
****************
    [  16  21  38  16  47  46  13  20  29   2  48  15  48 ...  22   0 ]
==== cpu scan, power-of-two ====
   elapsed time: 126.438ms    (std::chrono Measured)
    [   0  16  37  75  91 138 184 197 217 246 248 296 311 ... -2015557726 -2015557704 ]
==== cpu scan, non-power-of-two ====
   elapsed time: 133.342ms    (std::chrono Measured)
    [   0  16  37  75  91 138 184 197 217 246 248 296 311 ... -2015557816 -2015557801 ]
    passed
==== naive scan, power-of-two ====
   elapsed time: 197.473ms    (CUDA Measured)
    passed
==== naive scan, non-power-of-two ====
   elapsed time: 196.16ms    (CUDA Measured)
    passed
==== work-efficient scan, power-of-two ====
   elapsed time: 140.688ms    (CUDA Measured)
    passed
==== work-efficient scan, non-power-of-two ====
   elapsed time: 140.298ms    (CUDA Measured)
    passed
==== thrust scan, power-of-two ====
   elapsed time: 7.34115ms    (CUDA Measured)
    passed
==== thrust scan, non-power-of-two ====
   elapsed time: 7.69514ms    (CUDA Measured)
    passed

*****************************
** STREAM COMPACTION TESTS **
*****************************
    [   0   0   3   0   3   2   0   3   0   1   1   0   3 ...   0   0 ]
==== cpu compact without scan, power-of-two ====
   elapsed time: 610.962ms    (std::chrono Measured)
    [   3   3   2   3   1   1   3   2   3   2   2   3   2 ...   3   1 ]
    passed
==== cpu compact without scan, non-power-of-two ====
   elapsed time: 618.107ms    (std::chrono Measured)
    [   3   3   2   3   1   1   3   2   3   2   2   3   2 ...   3   1 ]
    passed
==== cpu compact with scan ====
   elapsed time: 960.001ms    (std::chrono Measured)
    [   3   3   2   3   1   1   3   2   3   2   2   3   2 ...   3   1 ]
    passed
==== work-efficient compact, power-of-two ====
   elapsed time: 165.221ms    (CUDA Measured)
    passed
==== work-efficient compact, non-power-of-two ====
   elapsed time: 164.069ms    (CUDA Measured)
    passed


18. arr size = 10,000

****************
** SCAN TESTS **
****************
    [  14   0  27  19  22  16  33  23   8   1   8  13  16 ...  28   0 ]
==== cpu scan, power-of-two ====
   elapsed time: 0.006ms    (std::chrono Measured)
    [   0  14  14  41  60  82  98 131 154 162 163 171 184 ... 244406 244434 ]
==== cpu scan, non-power-of-two ====
   elapsed time: 0.0041ms    (std::chrono Measured)
    [   0  14  14  41  60  82  98 131 154 162 163 171 184 ... 244338 244351 ]
    passed
==== naive scan, power-of-two ====
   elapsed time: 0.146432ms    (CUDA Measured)
    passed
==== naive scan, non-power-of-two ====
   elapsed time: 0.094208ms    (CUDA Measured)
    passed
==== work-efficient scan, power-of-two ====
   elapsed time: 0.26112ms    (CUDA Measured)
    passed
==== work-efficient scan, non-power-of-two ====
   elapsed time: 0.29184ms    (CUDA Measured)
    passed
==== thrust scan, power-of-two ====
   elapsed time: 0.103424ms    (CUDA Measured)
    passed
==== thrust scan, non-power-of-two ====
   elapsed time: 0.069632ms    (CUDA Measured)
    passed

*****************************
** STREAM COMPACTION TESTS **
*****************************
    [   2   0   3   1   1   2   3   2   3   1   0   3   1 ...   1   0 ]
==== cpu compact without scan, power-of-two ====
   elapsed time: 0.0325ms    (std::chrono Measured)
    [   2   3   1   1   2   3   2   3   1   3   1   3   2 ...   3   1 ]
    passed
==== cpu compact without scan, non-power-of-two ====
   elapsed time: 0.027ms    (std::chrono Measured)
    [   2   3   1   1   2   3   2   3   1   3   1   3   2 ...   2   3 ]
    passed
==== cpu compact with scan ====
   elapsed time: 0.0386ms    (std::chrono Measured)
    [   2   3   1   1   2   3   2   3   1   3   1   3   2 ...   3   1 ]
    passed
==== work-efficient compact, power-of-two ====
   elapsed time: 0.282624ms    (CUDA Measured)
    passed
==== work-efficient compact, non-power-of-two ====
   elapsed time: 0.372736ms    (CUDA Measured)
    passed


19. arr size = 1,000,000

****************
** SCAN TESTS **
****************
    [  38   5  38  21  10  41   2  40   6   0  37  37  43 ...  21   0 ]
==== cpu scan, power-of-two ====
   elapsed time: 0.4576ms    (std::chrono Measured)
    [   0  38  43  81 102 112 153 155 195 201 201 238 275 ... 24455654 24455675 ]
==== cpu scan, non-power-of-two ====
   elapsed time: 0.4375ms    (std::chrono Measured)
    [   0  38  43  81 102 112 153 155 195 201 201 238 275 ... 24455578 24455579 ]
    passed
==== naive scan, power-of-two ====
   elapsed time: 0.705472ms    (CUDA Measured)
    passed
==== naive scan, non-power-of-two ====
   elapsed time: 0.573632ms    (CUDA Measured)
    passed
==== work-efficient scan, power-of-two ====
   elapsed time: 0.720896ms    (CUDA Measured)
    passed
==== work-efficient scan, non-power-of-two ====
   elapsed time: 0.709632ms    (CUDA Measured)
    passed
==== thrust scan, power-of-two ====
   elapsed time: 0.574208ms    (CUDA Measured)
    passed
==== thrust scan, non-power-of-two ====
   elapsed time: 0.607904ms    (CUDA Measured)
    passed

*****************************
** STREAM COMPACTION TESTS **
*****************************
    [   2   3   2   3   0   3   0   0   0   2   3   1   3 ...   1   0 ]
==== cpu compact without scan, power-of-two ====
   elapsed time: 2.52ms    (std::chrono Measured)
    [   2   3   2   3   3   2   3   1   3   1   3   1   1 ...   2   1 ]
    passed
==== cpu compact without scan, non-power-of-two ====
   elapsed time: 2.7359ms    (std::chrono Measured)
    [   2   3   2   3   3   2   3   1   3   1   3   1   1 ...   3   3 ]
    passed
==== cpu compact with scan ====
   elapsed time: 4.4376ms    (std::chrono Measured)
    [   2   3   2   3   3   2   3   1   3   1   3   1   1 ...   2   1 ]
    passed
==== work-efficient compact, power-of-two ====
   elapsed time: 1.00077ms    (CUDA Measured)
    passed
==== work-efficient compact, non-power-of-two ====
   elapsed time: 1.06157ms    (CUDA Measured)
    passed
CUDA Stream Compaction
======================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 2**

* Zhenzhong Tang
  * [LinkedIn](https://www.linkedin.com/in/zhenzhong-anthony-tang-82334a210), [Instagram](https://instagram.com/toytag12), [personal website](https://toytag.net/)
* Tested on: Windows 11 Pro 22H2, AMD EPYC 7V12 64-Core Processor (4 vCPU cores) @ 2.44GHz 28GiB, Tesla T4 16GiB (Azure)

## Performance Analysis


### Sample Output with `int[2^29]`

```
****************
** SCAN TESTS **
****************
    [  39  29  12  48  27  43  42  11   9   8   5   1   5 ...  11   0 ]
==== cpu scan, power-of-two ====
   elapsed time: 1370.71ms    (std::chrono Measured)
    [   0  39  68  80 128 155 198 240 251 260 268 273 274 ... 263761477 263761488 ]
==== cpu scan, non-power-of-two ====
   elapsed time: 1369.71ms    (std::chrono Measured)
    [   0  39  68  80 128 155 198 240 251 260 268 273 274 ... 263761350 263761388 ]
    passed
==== naive scan, power-of-two ====
   elapsed time: 508.908ms    (CUDA Measured)
    passed
==== naive scan, non-power-of-two ====
   elapsed time: 506.161ms    (CUDA Measured)
    passed
==== work-efficient scan, power-of-two ====
   elapsed time: 48.7929ms    (CUDA Measured)
    passed
==== work-efficient scan, non-power-of-two ====
   elapsed time: 48.7834ms    (CUDA Measured)
    passed
==== thrust scan, power-of-two ====
   elapsed time: 17.1458ms    (CUDA Measured)
    passed
==== thrust scan, non-power-of-two ====
   elapsed time: 16.7649ms    (CUDA Measured)
    passed

*****************************
** STREAM COMPACTION TESTS **
*****************************
    [   3   0   0   0   1   3   1   3   3   1   3   0   1 ...   0   0 ]
==== cpu compact without scan, power-of-two ====
   elapsed time: 1345.11ms    (std::chrono Measured)
    [   3   1   3   1   3   3   1   3   1   3   2   3   3 ...   2   2 ]
    passed
==== cpu compact without scan, non-power-of-two ====
   elapsed time: 1300.07ms    (std::chrono Measured)
    [   3   1   3   1   3   3   1   3   1   3   2   3   3 ...   1   2 ]
    passed
==== cpu compact with scan ====
   elapsed time: 2031.72ms    (std::chrono Measured)
    [   3   1   3   1   3   3   1   3   1   3   2   3   3 ...   2   2 ]
    passed
==== work-efficient compact, power-of-two ====
   elapsed time: 183.325ms    (CUDA Measured)
    passed
==== work-efficient compact, non-power-of-two ====
   elapsed time: 180.469ms    (CUDA Measured)
    passed
```

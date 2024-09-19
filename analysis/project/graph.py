import matplotlib.pyplot as plt
import csv
import os
import statistics
from collections import defaultdict
import numpy as np

data_path = "../data"

scan_types = ["CPU", "Naive", "Efficient", "Thrust"]
array_sizes = []
ys = {}
errs = {}

for scan_type in scan_types:
    ys[scan_type] = []
    errs[scan_type] = []

filenames = [filename for filename in os.listdir(data_path)]
filenames.sort(key=lambda name: int(name.split('.')[0]))
# filenames = filenames[:-1]
# filenames = [filenames[-1]]

for filename in filenames:
    array_sizes.append(1 << int(filename.split('.')[0]))
    with open(data_path + "/" + filename) as file:
        reader = csv.reader(file)
        for (scan_type, line) in zip(scan_types, reader):
            contents = [float(val) for val in line if val != '']
            mean = statistics.mean(contents)
            stdev = statistics.stdev(contents)
            ys[scan_type].append(mean)
            errs[scan_type].append(stdev)

            print(array_sizes[-1], scan_type, mean, stdev)

# xs = np.arange(len(array_sizes))
# fig, ax = plt.subplots()
# bar_width = 0.1

# for (index, scan_type) in enumerate(scan_types):
#     ax.bar(
#         xs + (index - len(scan_types) / 2) * bar_width, ys[scan_type], bar_width, label = scan_type, yerr=errs[scan_type]
#     )

# ax.legend()

# # ax.set_ylim(0, 1)
# ax.set_ylabel("Milliseconds")

# # ax.set_xlabel("Array size")
# ax.set_xticks([])
# # ax.set_xticks(xs)
# # ax.set_xticklabels(array_sizes)

# ax.set_title("Time to run scan on array of 2^23")

# plt.savefig('chart.png')
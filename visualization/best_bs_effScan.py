import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("scan_size_results.csv")  # Make sure this filename matches yours

x_labels = [f'$2^{{{int(np.log2(n))}}}$' for n in df['N']]


plt.figure(figsize=(12, 6))
plt.bar(x_labels[1:8], df['Efficient(ms)'][1
                                           :8], color='green')


plt.xlabel('Input Size (N)')
plt.ylabel('Efficient GPU Scan Time (ms)')
plt.title('Efficient GPU Scan Runtime Across Input Sizes (Best Observed)')
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', linewidth=0.5)
plt.tight_layout()

plt.savefig('efficient_scan_bar_plot.png')
plt.show()

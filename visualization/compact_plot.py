import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


df = pd.read_csv("compact_size_results.csv")

plt.figure(figsize=(10, 6))
plt.yscale('log')

plt.plot(df['N'], df['CPU_NoScan(ms)'], label='CPU', marker='o')
plt.plot(df['N'], df['CPU_WithScan(ms)'], label='CPU with Scan', marker='o')
plt.plot(df['N'], df['GPU_Efficient(ms)'], label='Efficient GPU Scan', marker='o')


plt.xscale('log', base=2)
plt.xticks(df['N'], labels=[f'2^{int(np.log2(n))}' for n in df['N']], rotation=45)
plt.xlabel('Input Size (N)')
plt.ylabel('Elapsed Time (ms)')
plt.title('Compact Elapsed Time vs Input Size (Block Size = 256)')
plt.grid(True, which="both", linestyle='--', linewidth=0.5)
plt.legend()
plt.tight_layout()

plt.savefig('compact_size_plot.png')
plt.show()

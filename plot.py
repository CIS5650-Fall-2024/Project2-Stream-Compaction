import pandas as pd
import matplotlib.pyplot as plt
import os

# Load the Excel file
file_path = 'Performance.xlsx'  # Adjust the path if necessary
excel_data = pd.ExcelFile(file_path)

# Ensure the output directory exists
output_dir = 'img'
os.makedirs(output_dir, exist_ok=True)


# Function to process each sheet and plot the data
def plot_sheet(sheet_name, column_names, x_column, title, output_file):
    df = excel_data.parse(sheet_name)

    # Clean up the data by renaming columns appropriately
    df.columns = column_names
    df = df[1:].reset_index(drop=True)

    # Extract x and y values
    x_values = pd.to_numeric(df[x_column], errors='coerce')  # Using the "Power of 2" column for x-values
    y_values = {col: pd.to_numeric(df[col], errors='coerce') for col in column_names[1:]}  # Ignore non-numeric values

    # Plot the data with log scale for y-axis
    plt.figure(figsize=(10, 6))

    for label, y_data in y_values.items():
        plt.plot(x_values, y_data, label=label, marker='o')

    plt.xlabel('Array Size (Power of 2)')
    plt.ylabel('Time (ms)')
    plt.yscale('log')  # Set y-axis to logarithmic scale
    plt.title(title)
    plt.legend()
    plt.grid(True)

    # Save the plot to a file
    output_path = os.path.join(output_dir, output_file)
    plt.savefig(output_path)
    plt.close()


# Plot for the 'Scan' sheet
scan_columns = ['Power of 2', 'CPU Scan Time (ms)', 'Naive Scan Time (ms)', 'Efficient Scan Time (ms)', 'Thrust Scan Time (ms)']
plot_sheet('Scan', scan_columns, 'Power of 2', 'Scan Time vs Array Size', 'scan_time_vs_array_size.png')

# Plot for the 'Compaction' sheet
compaction_columns = ['Power of 2', 'CPU Compact without Scan Time (ms)', 'CPU Compact with Scan Time (ms)', 'Efficient GPU Compact Time (ms)']
plot_sheet('Compaction', compaction_columns, 'Power of 2', 'Compaction Time vs Array Size', 'compaction_time_vs_array_size.png')

print(f"Plots saved to {output_dir}")
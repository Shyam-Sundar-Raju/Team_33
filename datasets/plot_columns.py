import pandas as pd
import matplotlib.pyplot as plt
import os
import argparse

def plot_columns(file_path, output_dir, max_points=None):
    """
    Reads a dataset file and creates a point graph (scatter plot) for each column.
    """
    print(f"Reading {file_path}...")
    # Read the dataset (assuming CSV format based on train1.txt)
    df = pd.read_csv(file_path)
    
    # Optional subsampling to avoid memory/performance issues with huge datasets
    if max_points and len(df) > max_points:
        print(f"Subsampling to {max_points} points for faster plotting...")
        df = df.sample(n=max_points, random_state=42).sort_index()

    os.makedirs(output_dir, exist_ok=True)
    
    # Determine x-axis
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        x = df['timestamp']
    else:
        x = df.index

    # Plot each column
    for col in df.columns:
        if col == 'timestamp':
            continue
            
        print(f"Plotting {col}...")
        plt.figure(figsize=(10, 5))
        
        # Plot point graph (scatter plot)
        plt.plot(x, df[col], marker='.', linestyle='none', markersize=2, alpha=0.5)
        
        plt.title(f"Point Graph for {col}")
        plt.xlabel("Timestamp" if 'timestamp' in df.columns else "Index")
        plt.ylabel(col)
        plt.grid(True)
        plt.tight_layout()
        
        # Save the plot
        safe_col_name = str(col).replace("/", "_").replace("\\", "_")
        output_file = os.path.join(output_dir, f"{safe_col_name}.png")
        plt.savefig(output_file)
        plt.close()
        
    print(f"All plots saved to {output_dir}/")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot point graphs for each column in a dataset.")
    parser.add_argument("file_path", help="Path to the dataset file (e.g., datasets/hai/train1.txt)")
    parser.add_argument("--out", default="plots", help="Output directory for the plots")
    parser.add_argument("--max_points", type=int, default=10000, help="Maximum number of points to plot per column to prevent memory issues.")
    
    args = parser.parse_args()
    plot_columns(args.file_path, args.out, args.max_points)

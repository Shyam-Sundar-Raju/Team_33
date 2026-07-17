import pandas as pd
import argparse

def find_min_max_null(file_path):
    print(f"Reading {file_path}...")
    # Read the dataset
    df = pd.read_csv(file_path)
    
    # Select only numeric columns to avoid issues with timestamps or strings
    numeric_df = df.select_dtypes(include='number')
    
    results = []
    print("\n--- Min / Max / Null Values ---")
    for col in numeric_df.columns:
        col_min = numeric_df[col].min()
        col_max = numeric_df[col].max()
        
        # Check if there is at least one null value in the column
        has_null = 1 if df[col].isnull().any() else 0
        
        results.append({'Column': col, 'Min': col_min, 'Max': col_max, 'Null': has_null})
        print(f"{col}: Min = {col_min}, Max = {col_max}, Null = {has_null}")
        
    # Save the results to a CSV file for easy viewing later
    results_df = pd.DataFrame(results)
    
    # Extract just the filename to use as the output prefix
    base_name = file_path.split('/')[-1].split('.')[0]
    output_file = f"{base_name}_min_max_null.csv"
    
    results_df.to_csv(output_file, index=False)
    print(f"\nResults have also been saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find min, max, and null indicator for each numeric column in a dataset.")
    parser.add_argument("file_path", help="Path to the dataset file (e.g., datasets/hai/train1.txt)")
    
    args = parser.parse_args()
    find_min_max_null(args.file_path)

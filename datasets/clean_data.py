import pandas as pd
import argparse
import os

def clean_dataset(file_path, output_path=None, strategy='ffill'):
    print(f"Reading {file_path}...")
    df = pd.read_csv(file_path)
    
    initial_rows, initial_cols = df.shape
    print(f"Initial shape: {initial_rows} rows, {initial_cols} columns")
    
    # 1. Drop columns that are 100% missing (NaN)
    df.dropna(axis=1, how='all', inplace=True)
    dropped_cols = initial_cols - df.shape[1]
    if dropped_cols > 0:
        print(f"Dropped {dropped_cols} columns that were entirely NaN.")
        
    # 2. Handle remaining missing values
    null_count = df.isnull().sum().sum()
    print(f"Total missing values before cleaning: {null_count}")
    
    if strategy == 'ffill':
        # Forward fill is usually best for time-series sensor data
        print("Applying forward fill (propagates the last valid observation forward)...")
        df.fillna(method='ffill', inplace=True)
        # Any remaining NaNs at the very beginning can be backward filled or filled with 0
        df.fillna(method='bfill', inplace=True)
        df.fillna(0, inplace=True) 
    elif strategy == 'mean':
        print("Filling missing values with the column mean...")
        # Only fill numeric columns with mean
        numeric_cols = df.select_dtypes(include='number').columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
    elif strategy == 'drop':
        print("Dropping rows that contain any missing values...")
        df.dropna(axis=0, how='any', inplace=True)
    
    final_null_count = df.isnull().sum().sum()
    print(f"Total missing values after cleaning: {final_null_count}")
    print(f"Final shape: {df.shape[0]} rows, {df.shape[1]} columns")
    
    # Save the cleaned dataset
    if output_path:
        out_file = output_path
    else:
        base_name = file_path.split('/')[-1].split('.')[0]
        dir_name = os.path.dirname(file_path) or '.'
        out_file = os.path.join(dir_name, f"{base_name}_cleaned.csv")
        
    print(f"Saving cleaned data to {out_file}...")
    df.to_csv(out_file, index=False)
    print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean missing (null/NaN) values in a dataset.")
    parser.add_argument("file_path", help="Path to the dataset file")
    parser.add_argument("--output", help="Optional output path for the cleaned dataset", default=None)
    parser.add_argument("--strategy", choices=['ffill', 'mean', 'drop'], default='ffill', 
                        help="Strategy to handle missing values: 'ffill' (forward-fill, best for sensors), 'mean', or 'drop' rows.")
    
    args = parser.parse_args()
    clean_dataset(args.file_path, args.output, args.strategy)

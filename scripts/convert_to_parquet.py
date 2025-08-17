#!/usr/bin/env python3
"""
Convert CSV to optimized Parquet format for faster loading.
One-time conversion script for the Gene Search application.
"""

import pandas as pd
import time
import os

def convert_csv_to_parquet():
    csv_path = "real_l1000_data.csv"
    parquet_path = "real_l1000_data.parquet"
    
    print("Converting CSV to Parquet format...")
    print(f"Input: {csv_path}")
    print(f"Output: {parquet_path}")
    
    # Check if CSV exists
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found!")
        return False
    
    # Get original file size
    csv_size_mb = os.path.getsize(csv_path) / (1024 * 1024)
    print(f"Original CSV size: {csv_size_mb:.1f} MB")
    
    start_time = time.time()
    
    try:
        # Load with optimized data types (same as app)
        print("Loading CSV with optimized data types...")
        dtype_dict = {
            'drug_id': 'category',
            'cell_line': 'category', 
            'gene_symbol': 'category',
            'log2fc': 'float32',
            'pvalue': 'float32'
        }
        
        # Load in chunks to prevent memory issues
        chunk_size = 250000
        chunks = []
        
        for i, chunk in enumerate(pd.read_csv(csv_path, chunksize=chunk_size, dtype=dtype_dict)):
            print(f"   Processing chunk {i+1}...")
            chunk = chunk.dropna(subset=["log2fc", "pvalue"])
            chunks.append(chunk)
        
        print("Concatenating chunks...")
        df = pd.concat(chunks, ignore_index=True)
        
        print(f"Loaded {len(df):,} rows with {len(df.columns)} columns")
        print(f"Unique genes: {df['gene_symbol'].nunique():,}")
        print(f"Unique drugs: {df['drug_id'].nunique():,}")
        print(f"Unique cell lines: {df['cell_line'].nunique():,}")
        
        # Sort by gene_symbol for better cache locality
        print("Sorting by gene_symbol for optimal performance...")
        df = df.sort_values('gene_symbol')
        
        # Save as Parquet with Snappy compression
        print("Saving as Parquet...")
        df.to_parquet(parquet_path, compression='snappy', index=False)
        
        # Get new file size
        parquet_size_mb = os.path.getsize(parquet_path) / (1024 * 1024)
        
        load_time = time.time() - start_time
        compression_ratio = (csv_size_mb - parquet_size_mb) / csv_size_mb * 100
        
        print("Conversion completed successfully!")
        print(f"Conversion time: {load_time:.1f} seconds")
        print(f"Parquet size: {parquet_size_mb:.1f} MB")
        print(f"Compression: {compression_ratio:.1f}% smaller")
        print(f"Expected speedup: {csv_size_mb/parquet_size_mb:.1f}x faster loading")
        
        return True
        
    except Exception as e:
        print(f"Conversion failed: {e}")
        return False

if __name__ == "__main__":
    success = convert_csv_to_parquet()
    if success:
        print("\nReady to update gene_search_poc.py to use Parquet!")
        print("Next: Update the load_data function to use .parquet file")
    else:
        print("\nPlease check the error and try again")
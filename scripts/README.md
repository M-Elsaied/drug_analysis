# Development Scripts

This folder contains utility and development scripts used during the project development.

## Scripts

- `convert_to_parquet.py` - Converts CSV data to optimized Parquet format
- `transform_real_data.py` - Processes raw L1000 ZIP files into unified dataset
- `find_optimal_chunk.py` - Determines optimal chunk sizes for memory-efficient loading
- `test_transformation.py` - Test script for data transformation validation
- `run_optimized.bat` - Windows batch file for running with memory optimizations

## Usage

These scripts are primarily for development and data preparation. The main application (`drug_target_finder.py`) runs independently and doesn't require these scripts.
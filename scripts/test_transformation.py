import pandas as pd
import zipfile
import os
from pathlib import Path
import sys

# Import functions from the main script
sys.path.append('.')
from transform_real_data import extract_drug_name, parse_cell_line_from_filename, process_zip_file

def test_sample_files():
    """
    Test the transformation with a few sample files
    """
    input_dir = r"C:\Users\Mohamed\Desktop\github\Drug Analysis\drug_analysis\output_zips\output_zips"
    
    # Get first 3 ZIP files for testing
    zip_files = [f for f in os.listdir(input_dir) if f.endswith('.zip')][:3]
    
    print(f"Testing with {len(zip_files)} sample files:")
    for zip_file in zip_files:
        print(f"  - {zip_file}")
    
    all_test_data = []
    
    for zip_file in zip_files:
        zip_path = os.path.join(input_dir, zip_file)
        print(f"\nProcessing: {zip_file}")
        
        # Test drug name extraction
        drug_name = extract_drug_name(zip_file)
        print(f"  Drug name: {drug_name}")
        
        # Test ZIP file processing
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                csv_files = [f for f in zip_ref.namelist() if f.endswith('.csv')]
                print(f"  CSV files found: {len(csv_files)}")
                
                for csv_file in csv_files:
                    cell_line = parse_cell_line_from_filename(csv_file)
                    print(f"    - {csv_file} -> Cell line: {cell_line}")
        
        except Exception as e:
            print(f"  Error: {str(e)}")
            continue
        
        # Process the entire ZIP file
        file_data = process_zip_file(zip_path)
        if not file_data.empty:
            all_test_data.append(file_data)
            print(f"  Processed rows: {len(file_data)}")
            print(f"  Sample data:")
            print(file_data.head(3).to_string(index=False))
        else:
            print(f"  No data extracted")
    
    # Combine all test data
    if all_test_data:
        combined_df = pd.concat(all_test_data, ignore_index=True)
        
        print(f"\n=== TEST RESULTS ===")
        print(f"Total rows: {len(combined_df)}")
        print(f"Unique drugs: {combined_df['drug_id'].nunique()}")
        print(f"Unique cell lines: {combined_df['cell_line'].nunique()}")
        print(f"Unique genes: {combined_df['gene_symbol'].nunique()}")
        
        print(f"\nDrugs found: {list(combined_df['drug_id'].unique())}")
        print(f"Cell lines found: {list(combined_df['cell_line'].unique())}")
        
        # Save test output
        test_output = r"C:\Users\Mohamed\Desktop\github\Drug Analysis\drug_analysis\test_output.csv"
        combined_df.to_csv(test_output, index=False)
        print(f"\nTest output saved to: {test_output}")
        
        # Compare format with synthetic data
        print(f"\n=== FORMAT COMPARISON ===")
        synthetic_path = r"C:\Users\Mohamed\Desktop\github\Drug Analysis\drug_analysis\synthetic_l1000_demo.csv"
        if os.path.exists(synthetic_path):
            synthetic_df = pd.read_csv(synthetic_path, nrows=5)
            print("Synthetic data format:")
            print(synthetic_df.head().to_string(index=False))
            
            print("\nTransformed data format:")
            print(combined_df.head().to_string(index=False))
            
            print(f"\nColumn comparison:")
            print(f"Synthetic columns: {list(synthetic_df.columns)}")
            print(f"Transformed columns: {list(combined_df.columns)}")
        
        return True
    else:
        print("No data was successfully processed!")
        return False

if __name__ == "__main__":
    success = test_sample_files()
    if success:
        print("\n✅ Test completed successfully!")
    else:
        print("\n❌ Test failed!")
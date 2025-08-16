import pandas as pd
import zipfile
import os
import glob
import re
from pathlib import Path
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def extract_drug_name(zip_filename):
    """
    Extract and clean drug name from ZIP filename
    Example: 'abacavir_sulfate.zip' -> 'Abacavir Sulfate'
    """
    drug_name = Path(zip_filename).stem
    drug_name = drug_name.replace('_', ' ')
    drug_name = ' '.join(word.capitalize() for word in drug_name.split())
    return drug_name

def parse_cell_line_from_filename(csv_filename):
    """
    Extract cell line from CSV filename
    Example: 'ABACAVIR_SULFATE large_intestine HELA - ABACAVIR_SULFATE large_intestine HELA.csv' -> 'HELA'
    """
    basename = Path(csv_filename).stem
    
    # Pattern to extract cell line: looks for tissue type followed by cell line code
    patterns = [
        r'(\w+)\s+(\w+)\s+-',  # tissue cellline -
        r'(\w+)\s+(\w+)$',     # tissue cellline at end
    ]
    
    for pattern in patterns:
        match = re.search(pattern, basename)
        if match:
            return match.group(2)  # Return the cell line code
    
    # Fallback: try to extract last word before dash
    parts = basename.split(' - ')[0].split()
    if len(parts) >= 2:
        return parts[-1]
    
    logger.warning(f"Could not parse cell line from filename: {csv_filename}")
    return "Unknown"

def process_csv_data(csv_path, drug_id, cell_line):
    """
    Process CSV data and transform to target format
    """
    try:
        df = pd.read_csv(csv_path)
        
        # Check if required columns exist
        required_cols = ['Name_GeneSymbol', 'Value_LogDiffExp', 'Significance_pvalue']
        if not all(col in df.columns for col in required_cols):
            logger.warning(f"Missing required columns in {csv_path}")
            return pd.DataFrame()
        
        # Transform to target format
        result = pd.DataFrame({
            'drug_id': drug_id,
            'cell_line': cell_line,
            'gene_symbol': df['Name_GeneSymbol'],
            'log2fc': df['Value_LogDiffExp'],
            'pvalue': df['Significance_pvalue']
        })
        
        # Remove any rows with missing gene symbols
        result = result.dropna(subset=['gene_symbol'])
        
        return result
        
    except Exception as e:
        logger.error(f"Error processing {csv_path}: {str(e)}")
        return pd.DataFrame()

def process_zip_file(zip_path):
    """
    Process a single ZIP file and return transformed data
    """
    drug_name = extract_drug_name(os.path.basename(zip_path))
    all_data = []
    
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            csv_files = [f for f in zip_ref.namelist() if f.endswith('.csv')]
            
            for csv_file in csv_files:
                # Extract CSV to temporary location
                csv_content = zip_ref.read(csv_file)
                
                # Parse cell line from filename
                cell_line = parse_cell_line_from_filename(csv_file)
                
                # Create temporary file to read with pandas
                temp_csv_path = f"temp_{os.path.basename(csv_file)}"
                with open(temp_csv_path, 'wb') as temp_file:
                    temp_file.write(csv_content)
                
                # Process the CSV data
                csv_data = process_csv_data(temp_csv_path, drug_name, cell_line)
                if not csv_data.empty:
                    all_data.append(csv_data)
                
                # Clean up temporary file
                os.remove(temp_csv_path)
    
    except Exception as e:
        logger.error(f"Error processing ZIP file {zip_path}: {str(e)}")
    
    if all_data:
        return pd.concat(all_data, ignore_index=True)
    else:
        return pd.DataFrame()

def main():
    """
    Main function to process all ZIP files and create unified dataset
    """
    # Configuration
    input_dir = r"C:\Users\Mohamed\Desktop\github\Drug Analysis\drug_analysis\output_zips\output_zips"
    output_file = r"C:\Users\Mohamed\Desktop\github\Drug Analysis\drug_analysis\real_l1000_data.csv"
    batch_size = 50  # Process files in batches to manage memory
    
    # Get all ZIP files
    zip_files = glob.glob(os.path.join(input_dir, "*.zip"))
    total_files = len(zip_files)
    
    logger.info(f"Found {total_files} ZIP files to process")
    
    # Initialize output file with headers
    headers_written = False
    processed_count = 0
    
    # Process files in batches
    for i in tqdm(range(0, total_files, batch_size), desc="Processing batches"):
        batch_files = zip_files[i:i + batch_size]
        batch_data = []
        
        logger.info(f"Processing batch {i//batch_size + 1}/{(total_files + batch_size - 1)//batch_size}")
        
        # Process each file in the batch
        for zip_path in tqdm(batch_files, desc="Processing files in batch", leave=False):
            try:
                file_data = process_zip_file(zip_path)
                if not file_data.empty:
                    batch_data.append(file_data)
                processed_count += 1
                
            except Exception as e:
                logger.error(f"Failed to process {zip_path}: {str(e)}")
        
        # Write batch data to file
        if batch_data:
            batch_df = pd.concat(batch_data, ignore_index=True)
            
            # Write to CSV (append mode after first batch)
            mode = 'w' if not headers_written else 'a'
            header = not headers_written
            
            batch_df.to_csv(output_file, mode=mode, header=header, index=False)
            headers_written = True
            
            logger.info(f"Batch completed. Added {len(batch_df)} rows. Total processed: {processed_count}/{total_files}")
    
    logger.info(f"Processing complete! Output saved to: {output_file}")
    
    # Print summary statistics
    if os.path.exists(output_file):
        final_df = pd.read_csv(output_file)
        logger.info(f"Final dataset shape: {final_df.shape}")
        logger.info(f"Unique drugs: {final_df['drug_id'].nunique()}")
        logger.info(f"Unique cell lines: {final_df['cell_line'].nunique()}")
        logger.info(f"Unique genes: {final_df['gene_symbol'].nunique()}")

if __name__ == "__main__":
    main()
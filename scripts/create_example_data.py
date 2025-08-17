#!/usr/bin/env python3
"""
Create a lightweight synthetic dataset for DrugTargetFinder examples
"""

import pandas as pd
import numpy as np
import random

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

def create_synthetic_l1000_data():
    """Create synthetic L1000-style drug-gene interaction data"""
    
    # Define sample drugs (real drug names)
    drugs = [
        "Aspirin", "Metformin", "Doxorubicin", "Tamoxifen", "Paclitaxel",
        "Cisplatin", "Imatinib", "Erlotinib", "Bevacizumab", "Rituximab",
        "Insulin", "Warfarin", "Simvastatin", "Atorvastatin", "Lisinopril"
    ]
    
    # Define sample genes (real cancer-related genes)
    genes = [
        "TP53", "BRCA1", "BRCA2", "MYC", "EGFR", "CCND1", "RB1", "PTEN",
        "AKT1", "PIK3CA", "KRAS", "BRAF", "MDM2", "CDKN2A", "ATM"
    ]
    
    # Define sample cell lines (real cancer cell lines)
    cell_lines = [
        "HeLa", "MCF7", "A549", "PC3", "HCT116", "SW480", "PANC1", "U87MG"
    ]
    
    # Generate synthetic data
    data = []
    
    for drug in drugs:
        for gene in genes:
            for cell_line in cell_lines:
                # Generate realistic log2 fold changes (-3 to +3)
                log2fc = np.random.normal(0, 1.5)
                log2fc = np.clip(log2fc, -5, 5)  # Clip extreme values
                
                # Generate realistic p-values (more significant for larger effects)
                base_pvalue = np.random.uniform(0.001, 0.2)
                # Make p-values smaller for larger absolute fold changes
                effect_bonus = abs(log2fc) / 5.0
                pvalue = base_pvalue * (1 - effect_bonus * 0.8)
                pvalue = max(pvalue, 0.0001)  # Minimum p-value
                
                data.append({
                    'drug_id': drug,
                    'cell_line': cell_line,
                    'gene_symbol': gene,
                    'log2fc': round(log2fc, 3),
                    'pvalue': round(pvalue, 6)
                })
    
    return pd.DataFrame(data)

def main():
    """Create and save the synthetic dataset"""
    print("Creating synthetic L1000 dataset...")
    
    df = create_synthetic_l1000_data()
    
    # Print dataset statistics
    print(f"Dataset created with:")
    print(f"- {len(df):,} total records")
    print(f"- {df['drug_id'].nunique()} unique drugs")
    print(f"- {df['gene_symbol'].nunique()} unique genes")
    print(f"- {df['cell_line'].nunique()} unique cell lines")
    
    # Save as CSV
    output_path = "../data/example_dataset.csv"
    df.to_csv(output_path, index=False)
    print(f"\nDataset saved to: {output_path}")
    
    # Show sample data
    print("\nSample data:")
    print(df.head(10))
    
    print(f"\nFile size: {df.memory_usage(deep=True).sum() / 1024:.1f} KB")

if __name__ == "__main__":
    main()
# Data Directory

This directory contains example datasets for DrugTargetFinder.

## Files

### `example_dataset.csv`
A lightweight synthetic dataset for demonstration and testing purposes.

**Dataset Statistics:**
- **1,800** total drug-gene interaction records
- **15** unique drugs (real drug names like Aspirin, Metformin, Doxorubicin)
- **15** unique genes (cancer-related genes like TP53, BRCA1, MYC)
- **8** unique cell lines (real cancer cell lines like HeLa, MCF7, A549)
- **File size:** ~318 KB

**Use Cases:**
- Testing the DrugTargetFinder interface
- Understanding the data format and structure
- Demonstrating drug discovery workflows
- Local development and testing

## Data Format

The dataset follows the L1000 Connectivity Map format:

| Column | Description |
|--------|-------------|
| `drug_id` | Drug compound identifier/name |
| `cell_line` | Cancer cell line name |
| `gene_symbol` | Gene symbol (HGNC format) |
| `log2fc` | Log2 fold change value |
| `pvalue` | Statistical significance (p-value) |

## Production Data

The live DrugTargetFinder application uses the complete L1000 dataset hosted on Hugging Face Hub:
- **15.9M** drug-gene interaction records  
- **2,650** unique drug compounds
- **978** genes across **62** cancer cell lines
- **Data source:** https://huggingface.co/datasets/melsaied1/drug-finder

## Note

The large production dataset files are excluded from this repository to keep it lightweight. The application connects directly to the remote data source for real-time analysis.
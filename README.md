# 🧬 DrugTargetFinder

**Gene-Centric Drug Discovery Platform**

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-url.streamlit.app)

## Overview

DrugTargetFinder is a powerful web application that enables researchers to discover drugs that significantly affect specific target genes across multiple cancer cell lines. Built on top of the L1000 Connectivity Map dataset, this platform provides an intuitive interface for drug repurposing, biomarker discovery, and hypothesis generation.

## Features

### 🎯 **Gene-Centric Analysis**
- Search across 978 genes from L1000 dataset
- Analyze both upregulation and downregulation effects
- Real-time filtering with customizable thresholds

### 📊 **Advanced Visualization**
- Interactive heatmaps showing drug effects across cell lines
- Statistical summaries with consistency metrics
- Professional data tables with gradient highlighting

### ⚡ **High Performance**
- DuckDB-powered remote queries with zero memory loading
- Direct connection to Hugging Face Hub dataset
- Real-time search across 15.9M drug-gene interactions
- Eliminates memory crashes and enables concurrent usage

### 🔬 **Comprehensive Data**
- **15.9M** drug-gene interaction records
- **2,650** unique drug compounds
- **978** genes across **62** cancer cell lines
- Powered by L1000 Connectivity Map data

## Quick Start

### 🚀 **Online Access**
Visit the live application: [DrugTargetFinder on Streamlit Cloud](https://your-app-url.streamlit.app)

### 💻 **Local Installation**

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/drug_analysis.git
   cd drug_analysis
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   streamlit run drug_target_finder.py
   ```

4. **Access the app**
   Open your browser to `http://localhost:8501`

## Usage Guide

### 🧬 **Gene Selection**
1. Choose your target gene from the searchable dropdown
2. Get contextual information for well-known genes (TP53, BRCA1, etc.)

### 📈 **Configure Search Parameters**
- **Effect Direction**: Upregulate or downregulate gene expression
- **Fold Change Threshold (θ)**: Minimum effect magnitude
- **P-value Threshold (τ)**: Statistical significance cutoff  
- **Consistency Threshold (ρ)**: Minimum percentage of cell lines showing effect

### 🎯 **Analyze Results**
- View ranked drug candidates with consistency metrics
- Explore interactive heatmaps of drug effects
- Download results in multiple formats (CSV, TXT reports)

## Technical Architecture

### **Data Pipeline**
```
L1000 Data → Hugging Face Hub → DuckDB Remote Queries → Real-time Results
```

### **Performance Optimizations**
- **Zero Memory Loading**: DuckDB queries remote data without local storage
- **Cloud-Native**: Direct Hugging Face Hub integration
- **Smart Caching**: Streamlit's @st.cache_data for instant reloads
- **Scalable Architecture**: Handles concurrent users without crashes

### **Key Files**
- `drug_target_finder.py` - Main Streamlit application with DuckDB integration
- `requirements.txt` - Python dependencies (includes duckdb)
- `.claude.md` - Development documentation

### **Remote Data Source**
- **Hugging Face Hub**: [melsaied1/drug-finder](https://huggingface.co/datasets/melsaied1/drug-finder)
- **Direct URL**: `https://huggingface.co/datasets/melsaied1/drug-finder/resolve/main/real_l1000_data.parquet`
- **Size**: 151MB Parquet file (never downloaded locally)

## Data Format

The application expects data with the following columns:
- `drug_id`: Drug compound identifier
- `cell_line`: Cancer cell line name  
- `gene_symbol`: Gene symbol
- `log2fc`: Log2 fold change value
- `pvalue`: Statistical significance

## Contributing

We welcome contributions! Please see our [contributing guidelines](.claude.md) for development notes and architecture details.

## License

This project is licensed under the [MIT License](LICENSE).

## Citation

If you use DrugTargetFinder in your research, please cite:

```
DrugTargetFinder: A Gene-Centric Drug Discovery Platform
Built on L1000 Connectivity Map Data
```

## Support

- 📖 **Documentation**: See [development log](.claude.md) for technical details
- 🐛 **Issues**: Report bugs via GitHub Issues
- 💡 **Feature Requests**: Submit via GitHub Issues

---

**Built with ❤️ using Streamlit | Powered by L1000 Connectivity Map Data**
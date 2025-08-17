
# drug_target_finder.py
# DrugTargetFinder - Gene-Centric Drug Discovery Platform
# Main application for discovering drugs that affect specific target genes
import math
import os
import time
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import duckdb
from functools import lru_cache

st.set_page_config(
    page_title="DrugTargetFinder",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Main header with improved styling
st.markdown("""
<div style="text-align: center; padding: 1rem 0; background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); 
            border-radius: 10px; margin-bottom: 2rem; color: white;">
    <h1 style="margin: 0; font-size: 2.5rem; font-weight: 700;">
        🧬 DrugTargetFinder
    </h1>
    <p style="margin: 0.5rem 0 0 0; font-size: 1.2rem; opacity: 0.9;">
        Discover drugs that significantly affect your target gene across cancer cell lines
    </p>
</div>
""", unsafe_allow_html=True)

# Add brief explanation
with st.expander("🎯 What does this tool do?", expanded=False):
    st.markdown("""
    This tool helps you discover **drugs that affect specific genes** by analyzing gene expression data across multiple cancer cell lines.

    **How it works:**
    1. **Select a gene** you're interested in (e.g., tumor suppressor, oncogene)
    2. **Choose direction**: Find drugs that increase or decrease the gene's expression
    3. **Set thresholds**: Define how strong and consistent the effect should be
    4. **Get results**: See which drugs meet your criteria, with statistics and visualizations

    **Perfect for:** Drug repurposing, biomarker discovery, hypothesis generation, and understanding drug mechanisms.
    """)

@st.cache_resource
def get_duckdb_connection():
    """Initialize DuckDB connection - cached as a resource"""
    return duckdb.connect()

def execute_query_with_retry(conn, query, max_retries=3, base_delay=2):
    """Execute DuckDB query with exponential backoff retry logic"""
    for attempt in range(max_retries):
        try:
            result = conn.execute(query).fetchdf()
            return result
        except Exception as e:
            error_msg = str(e).lower()
            
            # Check if it's a rate limiting error
            if "429" in error_msg or "rate limit" in error_msg or "too many requests" in error_msg:
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)  # Exponential backoff
                    st.warning(f"⏳ Rate limit detected. Retrying in {delay} seconds... (Attempt {attempt + 1}/{max_retries})")
                    time.sleep(delay)
                    continue
                else:
                    st.error("❌ **Rate limit exceeded.** Please wait a few minutes and refresh the page.")
                    raise e
            else:
                # For other errors, don't retry
                st.error(f"❌ **Database Error:** {str(e)}")
                raise e
    
    return None

@st.cache_data(show_spinner=False)
def get_dataset_info():
    """Get basic dataset information using DuckDB"""
    
    # Create a progress container for better UX
    progress_container = st.container()
    
    with progress_container:
        st.markdown("### 🚀 Connecting to DrugTargetFinder Database")
        
        # Create columns for a nice layout
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            # Progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Fun facts during loading
            facts = [
                "🌐 Connecting to Hugging Face Hub dataset...",
                "🧬 Querying 978 genes across 62 cancer cell lines...",
                "💊 Accessing 2,650 unique drug compounds...",
                "⚡ DuckDB enables zero-memory loading!",
                "🔬 Data sourced from L1000 connectivity map project..."
            ]
            
            import time
            for i, fact in enumerate(facts):
                progress_bar.progress((i + 1) * 20)
                status_text.markdown(f"**{fact}**")
                time.sleep(0.2)  # Brief pause for effect
            
            status_text.markdown("**✅ Connection established! Ready for drug discovery...**")
            progress_bar.progress(100)
            
            # Get dataset statistics using DuckDB
            conn = get_duckdb_connection()
            
            # URL to the Hugging Face dataset
            data_url = "https://huggingface.co/datasets/melsaied1/drug-finder/resolve/main/real_l1000_data.parquet"
            
            # Get basic stats
            stats_query = f"""
            SELECT 
                COUNT(*) as total_records,
                COUNT(DISTINCT gene_symbol) as unique_genes,
                COUNT(DISTINCT drug_id) as unique_drugs,
                COUNT(DISTINCT cell_line) as unique_cell_lines
            FROM '{data_url}'
            """
            
            # Use retry logic for queries
            try:
                stats_df = execute_query_with_retry(conn, stats_query)
                stats = stats_df.iloc[0] if not stats_df.empty else None
                
                # Get gene list
                genes_query = f"""
                SELECT DISTINCT gene_symbol 
                FROM '{data_url}' 
                ORDER BY gene_symbol
                """
                
                genes_df = execute_query_with_retry(conn, genes_query)
                genes_list = genes_df['gene_symbol'].tolist() if not genes_df.empty else []
            except Exception as e:
                st.error("Failed to connect to database. Please refresh the page.")
                st.stop()
            
            # Clear progress indicators after a moment
            time.sleep(0.3)
            progress_container.empty()
    
    return {
        'total_records': stats[0],
        'unique_genes': stats[1], 
        'unique_drugs': stats[2],
        'unique_cell_lines': stats[3],
        'genes_list': genes_list
    }

@st.cache_data(show_spinner=False)
def query_gene_data(gene: str, selected_cell_lines=None):
    """Query data for specific gene using DuckDB - memory efficient"""
    conn = get_duckdb_connection()
    
    # URL to the Hugging Face dataset
    data_url = "https://huggingface.co/datasets/melsaied1/drug-finder/resolve/main/real_l1000_data.parquet"
    
    # Build WHERE clause
    where_conditions = [f"gene_symbol = '{gene}'"]
    
    if selected_cell_lines:
        cell_lines_str = "', '".join(selected_cell_lines)
        where_conditions.append(f"cell_line IN ('{cell_lines_str}')")
    
    where_clause = " AND ".join(where_conditions)
    
    # Query specific gene data
    query = f"""
    SELECT drug_id, cell_line, gene_symbol, log2fc, pvalue
    FROM '{data_url}'
    WHERE {where_clause}
    ORDER BY drug_id, cell_line
    """
    
    # Execute query with retry logic
    try:
        result = execute_query_with_retry(conn, query)
        return result
    except Exception as e:
        st.error("❌ Failed to fetch gene data. Please try again in a few minutes.")
        return pd.DataFrame()  # Return empty DataFrame on failure

@st.cache_data(show_spinner=False)
def get_cell_lines():
    """Get list of all cell lines"""
    conn = get_duckdb_connection()
    
    data_url = "https://huggingface.co/datasets/melsaied1/drug-finder/resolve/main/real_l1000_data.parquet"
    
    query = f"""
    SELECT DISTINCT cell_line 
    FROM '{data_url}' 
    ORDER BY cell_line
    """
    
    try:
        result_df = execute_query_with_retry(conn, query)
        return result_df['cell_line'].tolist() if not result_df.empty else []
    except Exception as e:
        st.error("❌ Failed to fetch cell lines. Using default.")
        return []

# Initialize database connection and get dataset info
try:
    # Get dataset information from Hugging Face Hub via DuckDB
    dataset_info = get_dataset_info()
    
    # Show success with dataset statistics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📊 Total Records", f"{dataset_info['total_records']:,}")
    with col2:
        st.metric("🧬 Genes", f"{dataset_info['unique_genes']:,}")
    with col3:
        st.metric("💊 Drugs", f"{dataset_info['unique_drugs']:,}")
    with col4:
        st.metric("🧪 Cell Lines", f"{dataset_info['unique_cell_lines']}")
    
    # Get gene list for selectbox
    genes_list = dataset_info['genes_list']
    
except Exception as e:
    st.error("🚫 **Service Temporarily Unavailable**")
    st.warning("""
    **The DrugTargetFinder service is experiencing high traffic or rate limiting from the data provider.**
    
    **What's happening:**
    - Hugging Face Hub is temporarily limiting requests
    - Too many users are accessing the service simultaneously
    
    **Please try:**
    - ⏳ Wait 5-10 minutes and refresh the page
    - 🔄 Clear your browser cache
    - 📧 Contact support if the issue persists
    
    **This is a temporary issue and will resolve automatically.**
    """)
    st.info("💡 **Tip:** The service works best during off-peak hours (early morning or late evening UTC).")
    st.stop()

# ----- Sidebar controls -----
st.sidebar.markdown("""
<div style="text-align: center; padding: 1rem; background: #f0f2f6; border-radius: 10px; margin-bottom: 1rem;">
    <h2 style="margin: 0; color: #1f77b4;">🔍 Search Configuration</h2>
    <p style="margin: 0.5rem 0 0 0; color: #666; font-size: 0.9rem;">
        Configure your drug discovery search
    </p>
</div>
""", unsafe_allow_html=True)

# Add help information with better styling
with st.sidebar.expander("💡 Quick Help Guide", expanded=False):
    st.markdown("""
    ### 🎯 **Gene Selection**
    Choose the gene you want to analyze for drug effects.
    
    ### 📈 **Effect Direction**  
    - **Upregulated**: Find drugs that increase gene expression
    - **Downregulated**: Find drugs that decrease gene expression
    
    ### ⚙️ **Statistical Thresholds**
    - **θ (Theta)**: Minimum fold-change magnitude (higher = stronger effect)
    - **τ (Tau)**: Maximum p-value for significance (lower = more confident)
    - **ρ (Rho)**: Minimum % of cell lines showing effect (higher = more consistent)
    
    ### 💡 **Pro Tips**
    - Start with default values for initial exploration
    - Decrease thresholds to find more drugs
    - Increase thresholds for high-confidence hits
    """)

# Gene selection with search
st.sidebar.markdown("### 🧬 **Gene Selection**")
default_gene = "CCND1" if "CCND1" in genes_list else genes_list[0]
gene = st.sidebar.selectbox(
    "Choose your target gene:",
    genes_list, 
    index=genes_list.index(default_gene),
    help="Type to search or scroll to find your gene of interest"
)

# Show gene info if it's a well-known gene
gene_info = {
    "RB1": "Tumor suppressor gene, retinoblastoma protein",
    "TP53": "The Guardian of the Genome, tumor suppressor",
    "BRCA1": "Breast cancer susceptibility gene 1",
    "EGFR": "Epidermal growth factor receptor, oncogene",
    "MYC": "Proto-oncogene, transcription factor",
    "CCND1": "Cyclin D1, cell cycle regulation, oncogene"
}
if gene in gene_info:
    st.sidebar.info(f"ℹ️ {gene}: {gene_info[gene]}")

st.sidebar.markdown("### 📈 **Effect Direction**")
direction = st.sidebar.radio(
    "Find drugs that:",
    ["Upregulate gene expression", "Downregulate gene expression"], 
    index=1,
    help="Choose the type of drug effect you're looking for"
)
# Simplify direction for processing
direction = "Upregulated" if "Upregulate" in direction else "Downregulated"

st.sidebar.markdown("### ⚙️ **Statistical Thresholds**")
st.sidebar.markdown("*Adjust these values to control search stringency*")

log2fc_thresh = st.sidebar.slider(
    "🎯 Fold Change Threshold (θ)", 
    0.2, 3.0, 2.0, 0.1,
    help="Minimum fold-change magnitude (2^θ fold change). Higher = stronger effect required."
)

pval_thresh = st.sidebar.slider(
    "📊 Significance Threshold (τ)", 
    0.0, 0.20, 0.05, 0.005,
    format="%.3f",
    help="Maximum p-value for statistical significance. Lower = more confident results."
)

consistency_pct = st.sidebar.slider(
    "🎯 Consistency Threshold (ρ)", 
    30, 100, 50, 5,
    format="%d%%",
    help="Minimum percentage of cell lines showing the effect. Higher = more reproducible."
)

# Advanced options with better styling
with st.sidebar.expander("🔬 **Advanced Options**"):
    st.markdown("**Cell Line Filtering** (Optional)")
    
    # Get cell lines list using DuckDB
    all_cell_lines = get_cell_lines()
    
    selected_cls = st.multiselect(
        "Restrict analysis to specific cell lines:",
        all_cell_lines,
        help="Leave empty to analyze all cell lines. Select specific ones to focus your analysis."
    )
    
    if selected_cls:
        st.success(f"✅ Analysis restricted to {len(selected_cls)} cell line(s)")
    else:
        st.info(f"ℹ️ Analyzing all {dataset_info['unique_cell_lines']} cell lines")

# ----- Query: gene-first, then thresholds -----
# Use DuckDB for memory-efficient querying
progress_text = st.empty()
progress_text.markdown("🔄 **Querying gene data from remote database...**")

with st.spinner("Fetching drug-gene interactions..."):
    # Get gene data using DuckDB (only loads relevant data into memory)
    df_gene = query_gene_data(gene, selected_cls)

progress_text.empty()

if df_gene.empty:
    st.info("Selected gene not found in the dataset or no data for selected cell lines.")
    st.stop()

# Apply filters and compute results efficiently
df_gene["significant"] = df_gene["pvalue"] <= pval_thresh

if direction == "Upregulated":
    df_gene["passes"] = (df_gene["log2fc"] >= log2fc_thresh) & df_gene["significant"]
else:
    df_gene["passes"] = (df_gene["log2fc"] <= -log2fc_thresh) & df_gene["significant"]

# Compute per-drug consistency
progress_text = st.empty()
progress_text.markdown("🔄 **Computing drug statistics...**")

with st.spinner("Analyzing drug-gene interactions..."):
    totals = df_gene.groupby("drug_id")["cell_line"].nunique().rename("n_total")
    passes = df_gene[df_gene["passes"]].groupby("drug_id")["cell_line"].nunique().rename("n_pass")

progress_text.empty()

stats = pd.concat([totals, passes], axis=1).fillna(0).astype(int)
# edge case: if n_total == 0 (shouldn't happen), drop
stats = stats[stats["n_total"] > 0]
# min_consistent depends on each drug's n_total
stats["min_consistent"] = np.ceil((consistency_pct/100.0) * stats["n_total"]).astype(int)
stats["pass_pct"] = (100.0 * stats["n_pass"] / stats["n_total"]).round(1)

# add mean log2fc for rows that passed (optional KPI)
mean_fc = (df_gene[df_gene["passes"]]
           .groupby("drug_id")["log2fc"]
           .mean()
           .rename("mean_log2fc_pass"))
stats = stats.join(mean_fc, how="left")

# final filtered drugs
matched = stats[stats["n_pass"] >= stats["min_consistent"]].copy()
matched = matched.sort_values(by=["pass_pct","mean_log2fc_pass"], ascending=[False, False])

# Results header with enhanced styling
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 1.5rem; background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%); 
            border-radius: 15px; margin: 2rem 0;">
    <h2 style="margin: 0; color: #2c3e50; font-size: 2rem;">🎯 Drug Discovery Results</h2>
</div>
""", unsafe_allow_html=True)

# Search parameters summary
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown(f"""
    ### 🔍 **Search Parameters**
    - **🧬 Target Gene:** `{gene}` {gene_info.get(gene, '') if gene in gene_info else ''}
    - **📈 Effect Direction:** {direction} gene expression
    - **⚙️ Thresholds:** θ≥{log2fc_thresh} | τ≤{pval_thresh} | ρ≥{consistency_pct}%
    """)

with col2:
    # Results summary metrics
    if matched.shape[0] > 0:
        success_rate = (matched.shape[0] / stats.shape[0] * 100) if stats.shape[0] > 0 else 0
        st.metric("🎯 Success Rate", f"{success_rate:.1f}%", f"{matched.shape[0]}/{stats.shape[0]} drugs")
    
# Results summary with better styling
result_col1, result_col2, result_col3 = st.columns(3)

with result_col1:
    st.metric("📊 **Candidate Drugs**", f"{stats.shape[0]:,}", "analyzed")

with result_col2:
    color = "normal" if matched.shape[0] > 0 else "inverse"
    st.metric("✅ **Matched Drugs**", f"{matched.shape[0]:,}", "meet criteria")

with result_col3:
    if matched.shape[0] > 0:
        avg_consistency = matched['pass_pct'].mean()
        st.metric("📈 **Avg Consistency**", f"{avg_consistency:.1f}%", "across cell lines")

if matched.empty:
    st.markdown("""
    <div style="text-align: center; padding: 2rem; background: #fff3cd; border: 2px solid #ffc107; 
                border-radius: 15px; margin: 2rem 0;">
        <h3 style="color: #856404; margin-bottom: 1rem;">🔍 No Drugs Found</h3>
        <p style="color: #856404; margin-bottom: 1.5rem; font-size: 1.1rem;">
            No drugs met your current criteria. Here are some suggestions:
        </p>
        <div style="text-align: left; max-width: 500px; margin: 0 auto;">
            <p><strong>💡 Try These Adjustments:</strong></p>
            <ul style="color: #856404;">
                <li>🎯 <strong>Lower the fold-change threshold</strong> (θ) to find weaker effects</li>
                <li>📊 <strong>Increase the p-value threshold</strong> (τ) to be less stringent</li>
                <li>🎯 <strong>Reduce the consistency percentage</strong> (ρ) to allow more variable effects</li>
                <li>📈 <strong>Try the opposite direction</strong> (upregulate ↔ downregulate)</li>
            </ul>
        </div>
    </div>
    """, unsafe_allow_html=True)
else:
    # Success message
    st.success(f"🎉 Found {matched.shape[0]} drugs that {direction.lower()} {gene} expression!")
    
    # Add explanation of table columns with better styling
    with st.expander("📊 **Understanding Your Results**", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### 📋 **Column Definitions**
            - **🏷️ drug_id**: Drug compound name/identifier
            - **📊 n_total**: Total cell lines tested for this drug
            - **✅ n_pass**: Cell lines meeting your criteria  
            - **🎯 min_consistent**: Minimum required (based on ρ threshold)
            - **📈 pass_pct**: Consistency percentage across cell lines
            """)
        
        with col2:
            st.markdown("""
            ### 💡 **Interpretation Guide**
            - **Higher pass_pct** = More reliable/consistent effect
            - **Higher n_total** = More comprehensive testing
            - **Strong mean_log2fc_pass** = Stronger gene expression change
            - **Top results** = Best combination of consistency and effect size
            """)

    # Enhanced dataframe display
    st.markdown("### 🏆 **Top Drug Candidates**")
    
    # Style the dataframe
    styled_df = matched.reset_index().style.format({
        'pass_pct': '{:.1f}%',
        'mean_log2fc_pass': '{:.2f}',
    }).background_gradient(subset=['pass_pct'], cmap='Greens')
    
    st.dataframe(styled_df, use_container_width=True)

    # ----- Enhanced Heatmap visualization -----
    st.markdown("---")
    st.markdown("### 🔥 **Drug Effect Heatmap**")
    st.markdown(f"*Visualizing {direction.lower()} effects of top drugs on {gene} across cell lines*")
    
    if len(matched) > 20:
        st.info(f"📊 Showing top 20 drugs (out of {len(matched)} total) for better visualization")
        display_drugs = matched.head(20).index.tolist()
    else:
        display_drugs = matched.index.tolist()
    
    # pivot for heatmap: keep only rows for matched drugs and gene
    sub = df_gene[df_gene["drug_id"].isin(display_drugs)].copy()

    # mask out non-significant entries (pvalue > thresh)
    sub.loc[sub["pvalue"] > pval_thresh, "log2fc"] = np.nan

    pivot = sub.pivot_table(index="drug_id",
                            columns="cell_line",
                            values="log2fc",
                            aggfunc="mean")  # mean in case duplicates

    # Sort rows by matched order and columns alphabetically
    pivot = pivot.reindex(index=display_drugs)
    pivot = pivot.reindex(sorted(pivot.columns), axis=1)

    if not pivot.empty:
        fig, ax = plt.subplots(figsize=(max(8, 0.8*len(pivot.columns)), max(6, 0.6*len(pivot.index))))
        
        # Enhanced heatmap styling
        sns.heatmap(pivot, ax=ax, cmap="RdBu_r", vmin=-3, vmax=3, center=0, 
                   linewidths=0.5, linecolor="white",
                   cbar_kws={"label": "log2 Fold Change", "shrink": 0.8}, 
                   annot=False, fmt='.2f')
        
        ax.set_xlabel("Cancer Cell Line", fontsize=12, fontweight='bold')
        ax.set_ylabel("Drug Compound", fontsize=12, fontweight='bold')
        ax.set_title(f"{gene} Expression Changes | {direction} Effects\n(Gray = non-significant, p > {pval_thresh})", 
                    fontsize=14, fontweight='bold', pad=20)
        
        # Rotate labels for better readability
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        plt.tight_layout()
        st.pyplot(fig)
    else:
        st.warning("⚠️ No data available for heatmap visualization")

    # Enhanced download section
    st.markdown("---")
    st.markdown("### 📥 **Export Results**")
    
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        dl = matched.reset_index()
        st.download_button(
            label="📊 Download Results (CSV)",
            data=dl.to_csv(index=False).encode("utf-8"),
            file_name=f"DrugTargetFinder_{gene}_{direction}_{len(matched)}drugs.csv",
            mime="text/csv",
            help="Download complete results table as CSV file"
        )
    
    with col2:
        # Create summary report
        summary_text = f"""DrugTargetFinder Analysis Report
        
Target Gene: {gene}
Effect Direction: {direction} gene expression
Analysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}

Search Parameters:
- Fold Change Threshold (θ): ≥{log2fc_thresh}
- P-value Threshold (τ): ≤{pval_thresh}
- Consistency Threshold (ρ): ≥{consistency_pct}%

Results Summary:
- Candidate Drugs Analyzed: {stats.shape[0]:,}
- Drugs Meeting Criteria: {matched.shape[0]:,}
- Success Rate: {(matched.shape[0]/stats.shape[0]*100) if stats.shape[0] > 0 else 0:.1f}%

Top 10 Drug Candidates:
{matched.head(10)[['n_total', 'n_pass', 'pass_pct', 'mean_log2fc_pass']].to_string()}

Generated by DrugTargetFinder - Gene-Centric Drug Discovery Platform
"""
        
        st.download_button(
            label="📋 Download Report (TXT)",
            data=summary_text.encode("utf-8"),
            file_name=f"DrugTargetFinder_Report_{gene}_{direction}.txt",
            mime="text/plain",
            help="Download summary report as text file"
        )

# Footer and additional information
st.markdown("---")

# Data insights section
with st.expander("🔬 **Raw Data Explorer**", expanded=False):
    st.markdown(f"### Raw data for {gene} gene analysis")
    st.markdown(f"*Showing all {len(df_gene):,} drug-gene interactions for detailed inspection*")
    
    # Add filter options for raw data
    col1, col2 = st.columns(2)
    with col1:
        show_significant_only = st.checkbox("Show only significant results (p ≤ τ)", value=False)
    with col2:
        show_passed_only = st.checkbox("Show only results meeting criteria", value=False)
    
    # Filter raw data based on selections
    display_data = df_gene.copy()
    if show_significant_only:
        display_data = display_data[display_data["significant"]]
    if show_passed_only:
        display_data = display_data[display_data["passes"]]
    
    st.dataframe(
        display_data.sort_values(by=["drug_id","cell_line"]),
        use_container_width=True,
        height=400
    )

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 2rem; background: #f8f9fa; border-radius: 10px; margin-top: 3rem;">
    <h4 style="margin: 0; color: #6c757d;">🧬 DrugTargetFinder</h4>
    <p style="margin: 0.5rem 0 0 0; color: #6c757d; font-size: 0.9rem;">
        Powered by L1000 Connectivity Map data • Built with Streamlit<br>
        Accelerating drug discovery through gene-centric analysis
    </p>
</div>
""", unsafe_allow_html=True)

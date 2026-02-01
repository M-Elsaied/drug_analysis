# extreme_hits_finder.py
# ExtremeHitsFinder - Discover the Most Extreme Drug-Gene Interactions
# Find drugs with the strongest effects on gene expression regardless of specific target
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import duckdb

st.set_page_config(
    page_title="ExtremeHitsFinder",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Main header with improved styling
st.markdown("""
<div style="text-align: center; padding: 1rem 0; background: linear-gradient(90deg, #f2709c 0%, #ff9472 100%);
            border-radius: 10px; margin-bottom: 2rem; color: white;">
    <h1 style="margin: 0; font-size: 2.5rem; font-weight: 700;">
        ⚡ ExtremeHitsFinder
    </h1>
    <p style="margin: 0.5rem 0 0 0; font-size: 1.2rem; opacity: 0.9;">
        Discover the most extreme drug-gene interactions across all targets
    </p>
</div>
""", unsafe_allow_html=True)

# Add brief explanation
with st.expander("🎯 What does this tool do?", expanded=False):
    st.markdown("""
    This tool finds **drug-gene interactions with the strongest effects** by analyzing the magnitude of gene expression changes.

    **How it works:**
    1. **Set thresholds**: Define minimum effect size and statistical significance
    2. **Choose direction**: Find upregulated, downregulated, or both types of effects
    3. **Filter cell lines**: Optionally restrict to specific cancer cell lines
    4. **Get results**: See the top N most extreme drug-gene interactions

    **Perfect for:** Identifying potent drug effects, finding strong biomarkers, and discovering unexpected drug-gene relationships.
    """)

# Constants
DATA_URL = "https://huggingface.co/datasets/melsaied1/drug-finder/resolve/main/real_l1000_data.parquet"


@st.cache_resource
def get_duckdb_connection():
    """Initialize DuckDB connection - cached as a resource"""
    return duckdb.connect()


@st.cache_data(show_spinner=False)
def get_cell_lines():
    """Get list of all cell lines"""
    conn = get_duckdb_connection()

    query = f"""
    SELECT DISTINCT cell_line
    FROM '{DATA_URL}'
    ORDER BY cell_line
    """

    result = conn.execute(query).fetchall()
    return [row[0] for row in result]


@st.cache_data(show_spinner=False)
def get_drugs():
    """Get list of all drugs"""
    conn = get_duckdb_connection()

    query = f"""
    SELECT DISTINCT drug_id
    FROM '{DATA_URL}'
    ORDER BY drug_id
    """

    result = conn.execute(query).fetchall()
    return [row[0] for row in result]


@st.cache_data(show_spinner=False)
def get_genes():
    """Get list of all genes"""
    conn = get_duckdb_connection()

    query = f"""
    SELECT DISTINCT gene_symbol
    FROM '{DATA_URL}'
    ORDER BY gene_symbol
    """

    result = conn.execute(query).fetchall()
    return [row[0] for row in result]


@st.cache_data(show_spinner=False)
def get_dataset_stats():
    """Get basic dataset statistics"""
    conn = get_duckdb_connection()

    query = f"""
    SELECT
        COUNT(*) as total_records,
        COUNT(DISTINCT gene_symbol) as unique_genes,
        COUNT(DISTINCT drug_id) as unique_drugs,
        COUNT(DISTINCT cell_line) as unique_cell_lines
    FROM '{DATA_URL}'
    """

    result = conn.execute(query).fetchone()
    return {
        'total_records': result[0],
        'unique_genes': result[1],
        'unique_drugs': result[2],
        'unique_cell_lines': result[3]
    }


# Maximum results to cache (covers all |log2fc| = 10 entries plus buffer)
MAX_CACHED_RESULTS = 10000


@st.cache_data(show_spinner=False)
def query_extreme_hits_base(min_abs_log2fc: float, pvalue_threshold: float,
                            direction: str, selected_drugs: tuple = None,
                            selected_genes: tuple = None, selected_cell_lines: tuple = None):
    """Query top extreme drug-gene interactions using DuckDB.

    Always fetches up to MAX_CACHED_RESULTS to maximize cache hits when user adjusts top_n slider.
    The top_n slicing happens after this cached query.
    """
    conn = get_duckdb_connection()

    # Build WHERE conditions
    where_conditions = [
        f"pvalue <= {pvalue_threshold}",
        f"ABS(log2fc) >= {min_abs_log2fc}"
    ]

    # Direction filter
    if direction == "Upregulated":
        where_conditions.append("log2fc > 0")
    elif direction == "Downregulated":
        where_conditions.append("log2fc < 0")
    # "Both" - no additional filter needed

    # Drug filter
    if selected_drugs:
        drugs_str = "', '".join(selected_drugs)
        where_conditions.append(f"drug_id IN ('{drugs_str}')")

    # Gene filter
    if selected_genes:
        genes_str = "', '".join(selected_genes)
        where_conditions.append(f"gene_symbol IN ('{genes_str}')")

    # Cell line filter
    if selected_cell_lines:
        cell_lines_str = "', '".join(selected_cell_lines)
        where_conditions.append(f"cell_line IN ('{cell_lines_str}')")

    where_clause = " AND ".join(where_conditions)

    query = f"""
    SELECT
        drug_id,
        gene_symbol,
        cell_line,
        log2fc,
        pvalue,
        ABS(log2fc) as abs_log2fc
    FROM '{DATA_URL}'
    WHERE {where_clause}
    ORDER BY ABS(log2fc) DESC, pvalue ASC
    LIMIT {MAX_CACHED_RESULTS}
    """

    result = conn.execute(query).fetchdf()
    return result


def query_extreme_hits(top_n: int, min_abs_log2fc: float, pvalue_threshold: float,
                       direction: str, selected_drugs: list = None,
                       selected_genes: list = None, selected_cell_lines: list = None):
    """Wrapper that fetches cached base results and slices to top_n."""
    # Convert lists to tuples for caching (lists aren't hashable)
    drugs_tuple = tuple(selected_drugs) if selected_drugs else None
    genes_tuple = tuple(selected_genes) if selected_genes else None
    cell_lines_tuple = tuple(selected_cell_lines) if selected_cell_lines else None

    # Get cached base results
    base_results = query_extreme_hits_base(min_abs_log2fc, pvalue_threshold,
                                            direction, drugs_tuple,
                                            genes_tuple, cell_lines_tuple)

    # Slice to requested top_n
    return base_results.head(top_n)


# Initialize and show dataset stats
try:
    with st.spinner("Connecting to database..."):
        stats = get_dataset_stats()

    # Show dataset statistics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📊 Total Records", f"{stats['total_records']:,}")
    with col2:
        st.metric("🧬 Genes", f"{stats['unique_genes']:,}")
    with col3:
        st.metric("💊 Drugs", f"{stats['unique_drugs']:,}")
    with col4:
        st.metric("🧪 Cell Lines", f"{stats['unique_cell_lines']}")

except Exception as e:
    st.error(f"❌ Failed to connect to database: {e}")
    st.info("Please check your internet connection and try refreshing the page.")
    st.stop()

# ----- Sidebar controls -----
st.sidebar.markdown("""
<div style="text-align: center; padding: 1rem; background: #f0f2f6; border-radius: 10px; margin-bottom: 1rem;">
    <h2 style="margin: 0; color: #f2709c;">⚡ Search Configuration</h2>
    <p style="margin: 0.5rem 0 0 0; color: #666; font-size: 0.9rem;">
        Configure your extreme hits search
    </p>
</div>
""", unsafe_allow_html=True)

# Help guide
with st.sidebar.expander("💡 Quick Help Guide", expanded=False):
    st.markdown("""
    ### 🔢 **Top N Results**
    How many extreme hits to return (sorted by effect magnitude).

    ### 📈 **Min |log2FC|**
    Minimum absolute fold-change. Higher values = stronger effects only.
    - 1.0 = 2-fold change
    - 2.0 = 4-fold change
    - 3.0 = 8-fold change

    ### 📊 **P-value Threshold**
    Maximum p-value for statistical significance.

    ### 🔄 **Direction**
    - **Both**: Show strongest effects regardless of direction
    - **Upregulated**: Only increased expression
    - **Downregulated**: Only decreased expression

    ### 💡 **Pro Tips**
    - Start with defaults to see the most extreme effects
    - Increase Min |log2FC| to find only the strongest hits
    - Use cell line filter for tissue-specific analysis
    """)

st.sidebar.markdown("### 🔢 **Result Settings**")

top_n = st.sidebar.slider(
    "Top N Results",
    min_value=10,
    max_value=10000,
    value=10000,
    step=10,
    help="Number of extreme hits to return (max 10,000)"
)

st.sidebar.markdown("### ⚙️ **Thresholds**")

min_abs_log2fc = st.sidebar.slider(
    "Min |log2FC| Magnitude",
    min_value=0.5,
    max_value=10.0,
    value=10.0,
    step=0.1,
    help="Minimum absolute log2 fold-change. 2.0 = 4-fold change. Max in dataset is 10."
)

pvalue_threshold = st.sidebar.slider(
    "P-value Threshold",
    min_value=0.001,
    max_value=0.20,
    value=0.05,
    step=0.001,
    format="%.3f",
    help="Maximum p-value for statistical significance."
)

st.sidebar.markdown("### 📈 **Effect Direction**")

direction = st.sidebar.radio(
    "Show effects that are:",
    ["Both", "Upregulated", "Downregulated"],
    index=0,
    help="Filter by direction of gene expression change"
)

# Advanced options
with st.sidebar.expander("🔬 **Advanced Options**"):
    st.markdown("**Filter by Drug** (Optional)")
    all_drugs = get_drugs()
    selected_drugs = st.multiselect(
        "Restrict to specific drugs:",
        all_drugs,
        help="Leave empty to search all drugs."
    )
    if selected_drugs:
        st.success(f"✅ Restricted to {len(selected_drugs)} drug(s)")

    st.markdown("**Filter by Gene** (Optional)")
    all_genes = get_genes()
    selected_genes = st.multiselect(
        "Restrict to specific genes:",
        all_genes,
        help="Leave empty to search all genes."
    )
    if selected_genes:
        st.success(f"✅ Restricted to {len(selected_genes)} gene(s)")

    st.markdown("**Filter by Cell Line** (Optional)")
    all_cell_lines = get_cell_lines()
    selected_cell_lines = st.multiselect(
        "Restrict to specific cell lines:",
        all_cell_lines,
        help="Leave empty to search all cell lines."
    )
    if selected_cell_lines:
        st.success(f"✅ Restricted to {len(selected_cell_lines)} cell line(s)")

# ----- Query and Display Results -----
st.markdown("---")

with st.spinner("🔍 Searching for extreme hits..."):
    df_results = query_extreme_hits(
        top_n=top_n,
        min_abs_log2fc=min_abs_log2fc,
        pvalue_threshold=pvalue_threshold,
        direction=direction,
        selected_drugs=selected_drugs if selected_drugs else None,
        selected_genes=selected_genes if selected_genes else None,
        selected_cell_lines=selected_cell_lines if selected_cell_lines else None
    )

# Results header
st.markdown("""
<div style="text-align: center; padding: 1.5rem; background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            border-radius: 15px; margin: 2rem 0;">
    <h2 style="margin: 0; color: #2c3e50; font-size: 2rem;">⚡ Extreme Hits Results</h2>
</div>
""", unsafe_allow_html=True)

# Search parameters summary
col1, col2 = st.columns([2, 1])

with col1:
    direction_text = "both directions" if direction == "Both" else f"{direction.lower()} only"
    drug_text = f"{len(selected_drugs)} selected" if selected_drugs else "all"
    gene_text = f"{len(selected_genes)} selected" if selected_genes else "all"
    cell_line_text = f"{len(selected_cell_lines)} selected" if selected_cell_lines else "all"
    st.markdown(f"""
    ### 🔍 **Search Parameters**
    - **🔢 Top N:** {top_n} results
    - **📈 Min |log2FC|:** ≥ {min_abs_log2fc}
    - **📊 P-value:** ≤ {pvalue_threshold}
    - **🔄 Direction:** {direction_text}
    - **💊 Drugs:** {drug_text}
    - **🧬 Genes:** {gene_text}
    - **🧪 Cell Lines:** {cell_line_text}
    """)

with col2:
    if not df_results.empty:
        max_effect = df_results['abs_log2fc'].max()
        st.metric("🏆 Max |log2FC|", f"{max_effect:.2f}")

# Results summary
result_col1, result_col2, result_col3 = st.columns(3)

with result_col1:
    st.metric("📊 **Results Found**", f"{len(df_results):,}")

with result_col2:
    if not df_results.empty:
        unique_drugs = df_results['drug_id'].nunique()
        st.metric("💊 **Unique Drugs**", f"{unique_drugs:,}")

with result_col3:
    if not df_results.empty:
        unique_genes = df_results['gene_symbol'].nunique()
        st.metric("🧬 **Unique Genes**", f"{unique_genes:,}")

if df_results.empty:
    st.markdown("""
    <div style="text-align: center; padding: 2rem; background: #fff3cd; border: 2px solid #ffc107;
                border-radius: 15px; margin: 2rem 0;">
        <h3 style="color: #856404; margin-bottom: 1rem;">🔍 No Extreme Hits Found</h3>
        <p style="color: #856404; margin-bottom: 1.5rem; font-size: 1.1rem;">
            No drug-gene interactions met your current criteria.
        </p>
        <div style="text-align: left; max-width: 500px; margin: 0 auto;">
            <p><strong>💡 Try These Adjustments:</strong></p>
            <ul style="color: #856404;">
                <li>🎯 <strong>Lower the Min |log2FC|</strong> threshold</li>
                <li>📊 <strong>Increase the P-value threshold</strong></li>
                <li>🔄 <strong>Try "Both" directions</strong></li>
                <li>🧪 <strong>Remove cell line restrictions</strong></li>
            </ul>
        </div>
    </div>
    """, unsafe_allow_html=True)
else:
    st.success(f"⚡ Found {len(df_results)} extreme drug-gene interactions!")

    # ----- Bar Chart: Top 20 Extreme Hits -----
    st.markdown("### 📊 **Top 20 Extreme Hits**")
    st.markdown("*Horizontal bars showing effect magnitude and direction*")

    # Take top 20 for visualization
    df_top20 = df_results.head(20).copy()
    df_top20['label'] = df_top20['drug_id'] + ' → ' + df_top20['gene_symbol']

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))

    # Color by direction: green for up, red for down
    colors = ['#2ecc71' if x > 0 else '#e74c3c' for x in df_top20['log2fc']]

    # Plot horizontal bars
    bars = ax.barh(range(len(df_top20)), df_top20['log2fc'], color=colors, edgecolor='white', linewidth=0.5)

    # Customize
    ax.set_yticks(range(len(df_top20)))
    ax.set_yticklabels(df_top20['label'], fontsize=9)
    ax.invert_yaxis()  # Highest at top
    ax.set_xlabel('log2 Fold Change', fontsize=12, fontweight='bold')
    ax.set_title('Top 20 Extreme Drug-Gene Interactions', fontsize=14, fontweight='bold', pad=20)
    ax.axvline(x=0, color='black', linewidth=0.8, linestyle='-')

    # Add gridlines
    ax.xaxis.grid(True, linestyle='--', alpha=0.3)
    ax.set_axisbelow(True)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2ecc71', edgecolor='white', label='Upregulated'),
        Patch(facecolor='#e74c3c', edgecolor='white', label='Downregulated')
    ]
    ax.legend(handles=legend_elements, loc='lower right')

    plt.tight_layout()
    st.pyplot(fig)

    # ----- Styled Results Table -----
    st.markdown("### 🏆 **All Extreme Hits**")

    # Understanding columns
    with st.expander("📊 **Understanding Your Results**", expanded=False):
        st.markdown("""
        - **drug_id**: Drug compound identifier
        - **gene_symbol**: Target gene affected
        - **cell_line**: Cancer cell line where effect was observed
        - **log2fc**: Log2 fold-change (positive = upregulated, negative = downregulated)
        - **pvalue**: Statistical significance of the effect
        - **abs_log2fc**: Absolute magnitude of the effect (used for ranking)
        """)

    # Style the dataframe with gradient on abs_log2fc
    styled_df = df_results.style.format({
        'log2fc': '{:.3f}',
        'pvalue': '{:.2e}',
        'abs_log2fc': '{:.3f}'
    }).background_gradient(subset=['abs_log2fc'], cmap='YlOrRd')

    st.dataframe(styled_df, use_container_width=True, height=400)

    # ----- Export Section -----
    st.markdown("---")
    st.markdown("### 📥 **Export Results**")

    col1, col2 = st.columns(2)

    with col1:
        csv_data = df_results.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📊 Download Results (CSV)",
            data=csv_data,
            file_name=f"ExtremeHits_top{top_n}_log2fc{min_abs_log2fc}.csv",
            mime="text/csv",
            help="Download complete results as CSV file"
        )

    with col2:
        # Create summary report
        summary_text = f"""ExtremeHitsFinder Analysis Report

Analysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}

Search Parameters:
- Top N Results: {top_n}
- Min |log2FC|: >= {min_abs_log2fc}
- P-value Threshold: <= {pvalue_threshold}
- Direction: {direction}
- Drugs: {', '.join(selected_drugs) if selected_drugs else 'All'}
- Genes: {', '.join(selected_genes) if selected_genes else 'All'}
- Cell Lines: {', '.join(selected_cell_lines) if selected_cell_lines else 'All'}

Results Summary:
- Total Extreme Hits: {len(df_results):,}
- Unique Drugs: {df_results['drug_id'].nunique():,}
- Unique Genes: {df_results['gene_symbol'].nunique():,}
- Max |log2FC|: {df_results['abs_log2fc'].max():.3f}
- Min |log2FC| (in results): {df_results['abs_log2fc'].min():.3f}

Top 10 Extreme Hits:
{df_results.head(10)[['drug_id', 'gene_symbol', 'cell_line', 'log2fc', 'pvalue']].to_string(index=False)}

Generated by ExtremeHitsFinder - Drug Effect Magnitude Discovery Platform
"""

        st.download_button(
            label="📋 Download Report (TXT)",
            data=summary_text.encode('utf-8'),
            file_name=f"ExtremeHits_Report_top{top_n}.txt",
            mime="text/plain",
            help="Download summary report as text file"
        )

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 2rem; background: #f8f9fa; border-radius: 10px; margin-top: 3rem;">
    <h4 style="margin: 0; color: #6c757d;">⚡ ExtremeHitsFinder</h4>
    <p style="margin: 0.5rem 0 0 0; color: #6c757d; font-size: 0.9rem;">
        Powered by L1000 Connectivity Map data • Built with Streamlit<br>
        Discovering the most potent drug-gene interactions
    </p>
</div>
""", unsafe_allow_html=True)

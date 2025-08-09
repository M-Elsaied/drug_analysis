
# gene_search_poc.py
import math
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from functools import lru_cache

st.set_page_config(layout="wide")
st.title("🔎 Gene-Centric Drug Search — PoC")
st.caption("Find drugs that significantly affect your target gene across multiple cancer cell lines")

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

@st.cache_data(show_spinner=False)
def load_data(csv_path: str):
    df = pd.read_csv(csv_path)
    # basic checks
    required = {"drug_id","cell_line","gene_symbol","log2fc","pvalue"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {missing}")
    # normalize types
    df["drug_id"] = df["drug_id"].astype(str)
    df["cell_line"] = df["cell_line"].astype(str)
    df["gene_symbol"] = df["gene_symbol"].astype(str)
    df["log2fc"] = pd.to_numeric(df["log2fc"], errors="coerce")
    df["pvalue"] = pd.to_numeric(df["pvalue"], errors="coerce")
    df = df.dropna(subset=["log2fc","pvalue"])
    return df

# Path to synthetic demo CSV (same folder as the app)
CSV_PATH = "synthetic_l1000_demo.csv"

try:
    df = load_data(CSV_PATH)
except Exception as e:
    st.error(f"Failed to load data: {e}")
    st.stop()

# ----- Sidebar controls -----
st.sidebar.header("🔍 Search Filters")

# Add help information
with st.sidebar.expander("ℹ️ How to use these filters"):
    st.markdown("""
    **Gene**: Select the target gene to analyze

    **Direction**: Choose if you want drugs that increase (upregulate) or decrease (downregulate) gene expression

    **Thresholds**: Set minimum requirements for:
    - **θ (Theta)**: Minimum fold-change magnitude
    - **τ (Tau)**: Maximum p-value for significance
    - **ρ (Rho)**: Minimum % of cell lines showing effect
    """)

# gene select with typeahead
genes = sorted(df["gene_symbol"].unique().tolist())
default_gene = "RB1" if "RB1" in genes else genes[0]
gene = st.sidebar.selectbox("🧬 Target Gene", genes, index=genes.index(default_gene),
                           help="Select the gene you want to analyze for drug effects")

direction = st.sidebar.radio("📈 Effect Direction", ["Upregulated","Downregulated"], horizontal=True, index=1,
                            help="Choose whether you want drugs that increase or decrease gene expression")

st.sidebar.markdown("### 📊 Statistical Thresholds")
log2fc_thresh = st.sidebar.slider("log₂ Fold Change Threshold (θ)", 0.2, 3.0, 1.0, 0.1,
                                 help="Minimum fold-change magnitude required (higher = more stringent)")
pval_thresh = st.sidebar.slider("P-value Threshold (τ)", 0.0, 0.20, 0.15, 0.005,
                               help="Maximum p-value for statistical significance (lower = more stringent)")
consistency_pct = st.sidebar.slider("Consistency across cell lines (ρ, %)", 30, 100, 50, 5,
                                   help="Minimum percentage of cell lines that must show the effect")

# optional: subset of cell lines to consider
with st.sidebar.expander("Advanced"):
    selected_cls = st.multiselect("Restrict to cell lines (optional)", sorted(df["cell_line"].unique().tolist()))
if selected_cls:
    df = df[df["cell_line"].isin(selected_cls)]

# ----- Query: gene-first, then thresholds -----
df_gene = df[df["gene_symbol"] == gene].copy()
if df_gene.empty:
    st.info("Selected gene not found in the dataset.")
    st.stop()

# apply p-value filter first
df_gene["significant"] = df_gene["pvalue"] <= pval_thresh

if direction == "Upregulated":
    df_gene["passes"] = (df_gene["log2fc"] >= log2fc_thresh) & df_gene["significant"]
else:
    df_gene["passes"] = (df_gene["log2fc"] <= -log2fc_thresh) & df_gene["significant"]

# compute per-drug consistency
# n_total: number of distinct cell lines measured for this (drug, gene)
totals = df_gene.groupby("drug_id")["cell_line"].nunique().rename("n_total")
passes = df_gene[df_gene["passes"]].groupby("drug_id")["cell_line"].nunique().rename("n_pass")

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

st.subheader("Results")
st.write(f"**Gene:** `{gene}` | **Direction:** `{direction}` | θ={log2fc_thresh} | τ={pval_thresh} | ρ={consistency_pct}%")
st.write(f"""
- Total candidate drugs: {stats.shape[0]}
- Matched drugs: {matched.shape[0]}
""")

if matched.empty:
    st.warning("No drugs met the criteria. Try relaxing thresholds or changing the direction.")
else:
    # Add explanation of table columns
    with st.expander("📊 Understanding the Results Table", expanded=False):
        st.markdown("""
        **Column Explanations:**

        - **drug_id**: Name/identifier of the drug compound
        - **n_total**: Total number of cell lines tested for this drug-gene combination
        - **n_pass**: Number of cell lines that met your criteria (significant p-value + fold-change threshold)
        - **min_consistent**: Minimum number of cell lines required to pass (based on your consistency percentage)
        - **pass_pct**: Percentage of cell lines showing the desired effect (higher = more consistent)
        - **mean_log2fc_pass**: Average fold-change across cell lines that passed your criteria

        **How to interpret:**
        - Higher **pass_pct** = more reliable/consistent drug effect
        - More negative **mean_log2fc_pass** = stronger downregulation (for downregulated searches)
        - Higher **n_total** = drug tested in more cell lines (more comprehensive data)
        """)

    st.dataframe(matched.reset_index())

    # ----- Heatmap: drugs × cell lines (masked by p-value threshold) -----
    st.subheader("Heatmap (log2FC values)")
    # pivot for heatmap: keep only rows for matched drugs and gene
    keep_drugs = matched.index.tolist()
    sub = df_gene[df_gene["drug_id"].isin(keep_drugs)].copy()

    # mask out non-significant entries (pvalue > thresh)
    sub.loc[sub["pvalue"] > pval_thresh, "log2fc"] = np.nan

    pivot = sub.pivot_table(index="drug_id",
                            columns="cell_line",
                            values="log2fc",
                            aggfunc="mean")  # mean in case duplicates

    # Sort rows by matched order and columns alphabetically
    pivot = pivot.reindex(index=keep_drugs)
    pivot = pivot.reindex(sorted(pivot.columns), axis=1)

    fig, ax = plt.subplots(figsize=(max(6, 0.7*len(pivot.columns)), max(4, 0.5*len(pivot.index))))
    sns.heatmap(pivot, ax=ax, cmap="RdBu_r", vmin=-3, vmax=3, center=0, linewidths=0.3, linecolor="#e5e7eb",
                cbar_kws={"label": "log2FC"}, annot=False)
    ax.set_xlabel("Cell Line")
    ax.set_ylabel("Drug")
    ax.set_title(f"{gene} — {direction} (masked where p > {pval_thresh})")
    st.pyplot(fig)

    # download filtered results
    dl = matched.reset_index()
    st.download_button("Download matched drugs (CSV)",
                       data=dl.to_csv(index=False).encode("utf-8"),
                       file_name=f"matched_drugs_{gene}_{'up' if direction=='Upregulated' else 'down'}.csv",
                       mime="text/csv")

with st.expander("Debug / Peek at raw rows for selected gene"):
    st.dataframe(df_gene.sort_values(by=["drug_id","cell_line"]))

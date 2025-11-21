# src/eda.py
"""
Advanced EDA module for the Ultra-Processed Food Classification project.

Includes:
- Full data structure analysis
- Raw + binary target distribution
- Histograms, log-histograms, boxplots
- Correlation heatmap
- Missing-value heatmap
- Pairplot (sampled)
- Jointplots for key nutrient pairs
- Violin plots by class
- Top-frequency bar charts for categorical features
- Z-score outlier detection per numeric feature
- Cramér's V correlation for categorical variables
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore, chi2_contingency

TARGET_COL = "f_FPro_class"

RAW_CATEGORICAL = ["name", "store", "food category", "brand"]

NUMERIC_CANDIDATES = [
    "price", "price percal", "package_weight",
    "Protein", "Total Fat", "Carbohydrate", "Sugars, total",
    "Fiber, total dietary", "Calcium", "Iron", "Sodium",
    "Cholesterol", "Fatty acids, total saturated",
]

def binary_map(series: pd.Series) -> pd.Series:
    return series.map(lambda x: 0 if x == 3 else 1)

def cramers_v(col1, col2):
    """Compute Cramér's V for categorical associations."""
    confusion = pd.crosstab(col1, col2)
    chi2 = chi2_contingency(confusion)[0]
    n = confusion.sum().sum()
    phi2 = chi2 / n
    r, k = confusion.shape
    phi2corr = max(0, phi2 - ((k - 1)*(r - 1))/(n - 1))
    rcorr = r - ((r - 1)**2 / (n - 1))
    kcorr = k - ((k - 1)**2 / (n - 1))
    return np.sqrt(phi2corr / min((kcorr - 1), (rcorr - 1)))

def run_eda(df: pd.DataFrame, outdir: str = "reports") -> dict:
    os.makedirs(outdir, exist_ok=True)
    results = {}

    # ============================================================
    # 1. DATA STRUCTURE
    # ============================================================
    info_tbl = pd.DataFrame({
        "column": df.columns,
        "dtype": [df[c].dtype for c in df.columns],
        "n_missing": [df[c].isna().sum() for c in df.columns],
        "pct_missing": [df[c].isna().mean() * 100 for c in df.columns],
        "n_unique": [df[c].nunique(dropna=True) for c in df.columns],
    })
    info_tbl.to_csv(f"{outdir}/01_structure_columns.csv", index=False)
    results["structure"] = info_tbl

    numeric_cols = [c for c in NUMERIC_CANDIDATES if c in df.columns]

    # ============================================================
    # 2. TARGET DISTRIBUTION
    # ============================================================
    y_raw = df[TARGET_COL]
    y_bin = binary_map(y_raw)

    raw_counts = y_raw.value_counts().sort_index()
    raw_counts.to_csv(f"{outdir}/02_target_raw_distribution.csv", header=["count"])

    plt.figure(figsize=(6,4))
    sns.barplot(x=raw_counts.index.astype(str), y=raw_counts.values)
    plt.title("Raw Target Distribution")
    plt.tight_layout()
    plt.savefig(f"{outdir}/02_target_raw_distribution.png")
    plt.close()

    bin_counts = y_bin.value_counts().sort_index()
    bin_counts.to_csv(f"{outdir}/02_target_binary_distribution.csv", header=["count"])

    plt.figure(figsize=(6,4))
    sns.barplot(x=bin_counts.index.astype(str), y=bin_counts.values)
    plt.title("Binary Target Distribution")
    plt.tight_layout()
    plt.savefig(f"{outdir}/02_target_binary_distribution.png")
    plt.close()

    # ============================================================
    # 3. DESCRIPTIVE STATISTICS
    # ============================================================
    desc_stats = df[numeric_cols].describe().T
    desc_stats.to_csv(f"{outdir}/03_descriptive_stats_overall.csv")

    df_tmp = df.copy()
    df_tmp["binary_target"] = y_bin
    by_class = df_tmp.groupby("binary_target")[numeric_cols].describe().T
    by_class.to_csv(f"{outdir}/03_descriptive_stats_by_class.csv")

    # ============================================================
    # 4. HISTOGRAMS + LOG HISTOGRAMS
    # ============================================================
    for col in numeric_cols:
        # Standard distribution
        plt.figure(figsize=(6,4))
        sns.histplot(df[col], bins=30, kde=True)
        plt.title(f"Histogram: {col}")
        plt.tight_layout()
        plt.savefig(f"{outdir}/hist_{col.replace(' ','_')}.png")
        plt.close()

        # Log distribution (skip if <=0 present)
        if (df[col] > 0).sum() > 10:
            plt.figure(figsize=(6,4))
            sns.histplot(np.log1p(df[col]), bins=30, kde=True)
            plt.title(f"Log-Histogram: {col}")
            plt.tight_layout()
            plt.savefig(f"{outdir}/loghist_{col.replace(' ','_')}.png")
            plt.close()

    # ============================================================
    # 5. BOXPLOTS & VIOLIN PLOTS
    # ============================================================
    for col in numeric_cols:
        # Boxplot
        plt.figure(figsize=(6,4))
        sns.boxplot(x=df_tmp["binary_target"], y=df_tmp[col])
        plt.title(f"Boxplot by class: {col}")
        plt.tight_layout()
        plt.savefig(f"{outdir}/box_{col.replace(' ','_')}.png")
        plt.close()

        # Violin
        plt.figure(figsize=(6,4))
        sns.violinplot(x=df_tmp["binary_target"], y=df_tmp[col], inner="quartile")
        plt.title(f"Violin plot by class: {col}")
        plt.tight_layout()
        plt.savefig(f"{outdir}/violin_{col.replace(' ','_')}.png")
        plt.close()

    # ============================================================
    # 6. CORRELATION HEATMAP
    # ============================================================
    corr = df[numeric_cols].corr()
    plt.figure(figsize=(10,8))
    sns.heatmap(corr, cmap="coolwarm", annot=False)
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(f"{outdir}/04_correlation_heatmap.png")
    plt.close()

    # ============================================================
    # 7. MISSING VALUES HEATMAP
    # ============================================================
    plt.figure(figsize=(12,6))
    sns.heatmap(df.isna(), cbar=False)
    plt.title("Missing Value Heatmap")
    plt.tight_layout()
    plt.savefig(f"{outdir}/04_missing_values_heatmap.png")
    plt.close()

    # ============================================================
    # 8. TOP CATEGORICAL FREQUENCIES
    # ============================================================
    for col in RAW_CATEGORICAL:
        if col in df.columns:
            plt.figure(figsize=(10,5))
            df[col].value_counts(dropna=False).head(20).plot(kind="bar")
            plt.title(f"Top 20 categories: {col}")
            plt.tight_layout()
            plt.savefig(f"{outdir}/05_topfreq_{col.replace(' ','_')}.png")
            plt.close()

    # ============================================================
    # 9. PAIRPLOT (sampled to avoid huge files)
    # ============================================================
    sample_df = df_tmp.sample(min(400, len(df_tmp)), random_state=42)
    try:
        sns.pairplot(sample_df[numeric_cols[:6] + ["binary_target"]], hue="binary_target")
        plt.savefig(f"{outdir}/06_pairplot_sample.png")
        plt.close()
    except Exception:
        pass

    # ============================================================
    # 10. JOINT PLOTS (important nutrient comparisons)
    # ============================================================
    joint_pairs = [
        ("Protein", "Total Fat"),
        ("Carbohydrate", "Sugars, total"),
        ("Sodium", "Cholesterol"),
    ]

    for x, y in joint_pairs:
        if x in df.columns and y in df.columns:
            sns.jointplot(data=df_tmp, x=x, y=y, kind="kde", hue="binary_target")
            plt.savefig(f"{outdir}/06_jointplot_{x}_{y}.png")
            plt.close()

    # ============================================================
    # 11. Z-SCORE OUTLIER ANALYSIS
    # ============================================================
    z_df = df[numeric_cols].apply(zscore)
    outlier_flags = (np.abs(z_df) > 3).sum(axis=1)
    df_outliers = pd.DataFrame({
        "row_index": df.index,
        "n_outlier_features": outlier_flags
    })
    df_outliers.to_csv(f"{outdir}/07_zscore_outliers.csv", index=False)

    # ============================================================
    # 12. CRAMER'S V for categorical association
    # ============================================================
    cat_results = []
    cats = [c for c in RAW_CATEGORICAL if c in df.columns]

    for c1 in cats:
        for c2 in cats:
            if c1 < c2:  # avoid duplicates
                v = cramers_v(df[c1].astype(str), df[c2].astype(str))
                cat_results.append({"col1": c1, "col2": c2, "cramers_v": v})

    pd.DataFrame(cat_results).to_csv(f"{outdir}/08_cramers_v.csv", index=False)

    print(f"[EDA] Advanced EDA complete. Outputs saved to {outdir}.")
    return results

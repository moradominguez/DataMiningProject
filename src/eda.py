# src/eda.py
"""
EDA module for the Ultra-Processed Food Classification project.

Tasks covered:
- 1.1 Data Structure
- 1.2 Target Distribution (raw + binary)
- 1.3 Descriptive Statistics (overall + by binary class)
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

TARGET_COL = "f_FPro_class"

RAW_CATEGORICAL = ["name", "store", "food category", "brand"]

NUMERIC_CANDIDATES = [
    "price", "price per cal", "package_weight",
    "Protein", "Total Fat", "Carbohydrate", "Sugars, total",
    "Fiber, total dietary", "Calcium", "Iron", "Sodium",
    "Cholesterol", "Fatty acids, total saturated",
]


def binary_map(series: pd.Series) -> pd.Series:
    """Map f_FPro_class -> binary target (1 = non-UPF, 0 = UPF)."""
    return series.map(lambda x: 0 if x == 3 else 1)


def run_eda(df: pd.DataFrame, outdir: str = "reports") -> dict:
    """Run full EDA and save results in a flat reports folder."""
    os.makedirs(outdir, exist_ok=True)
    results = {}

    # ───────── 1.1 DATA STRUCTURE ─────────
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
    results["numeric_cols"] = numeric_cols

    # ───────── 1.2 TARGET DISTRIBUTION ─────────
    if TARGET_COL not in df.columns:
        raise ValueError(f"Missing target column: {TARGET_COL}")

    y_raw = df[TARGET_COL]
    y_bin = binary_map(y_raw)

    # raw distribution
    raw_counts = y_raw.value_counts(dropna=False).sort_index()
    raw_counts.to_csv(f"{outdir}/02_target_raw_distribution.csv", header=["count"])
    results["target_raw"] = raw_counts

    plt.figure(figsize=(6, 4))
    sns.barplot(x=raw_counts.index.astype(str), y=raw_counts.values)
    plt.title("Raw target distribution (f_FPro_class)")
    plt.xlabel("Class (0–3)"); plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(f"{outdir}/02_target_raw_distribution.png")
    plt.close()

    # binary distribution
    bin_counts = y_bin.value_counts(dropna=False).sort_index()
    bin_counts.to_csv(f"{outdir}/02_target_binary_distribution.csv", header=["count"])
    results["target_binary"] = bin_counts

    plt.figure(figsize=(6, 4))
    sns.barplot(x=bin_counts.index.astype(str), y=bin_counts.values)
    plt.title("Binary target distribution (1=non-UPF, 0=UPF)")
    plt.xlabel("Binary class"); plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(f"{outdir}/02_target_binary_distribution.png")
    plt.close()

    # ───────── 1.3 DESCRIPTIVE STATS ─────────
    # overall stats
    desc_stats = df[numeric_cols].describe().T
    desc_stats.to_csv(f"{outdir}/03_descriptive_stats_overall.csv")
    results["descriptive_overall"] = desc_stats

    # by binary class stats
    df_tmp = df.copy()
    df_tmp["binary_target"] = y_bin
    by_class = df_tmp.groupby("binary_target")[numeric_cols].describe().T
    by_class.to_csv(f"{outdir}/03_descriptive_stats_by_class.csv")
    results["descriptive_by_class"] = by_class

    # histograms
    for col in numeric_cols:
        plt.figure(figsize=(6, 4))
        sns.histplot(df[col], bins=30)
        plt.title(f"Histogram: {col}")
        plt.tight_layout()
        plt.savefig(f"{outdir}/hist_{col.replace(' ', '_')}.png")
        plt.close()

    # boxplots by binary class
    for col in numeric_cols:
        plt.figure(figsize=(6, 4))
        sns.boxplot(x=df_tmp["binary_target"], y=df_tmp[col])
        plt.title(f"Boxplot by binary class: {col}")
        plt.xlabel("Binary target (0=UPF, 1=non-UPF)")
        plt.tight_layout()
        plt.savefig(f"{outdir}/box_{col.replace(' ', '_')}.png")
        plt.close()

    print(f"[EDA] Completed. Reports saved to: {outdir}")
    return results

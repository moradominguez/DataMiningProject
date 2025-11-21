"""
Simple, clean and professor-aligned EDA module for the Ultra-Processed Food Classification project.

Generates:
- Data structure overview
- Raw + binary target distribution
- Numeric stats (overall + by class)
- Categorical summaries
- Histograms
- Boxplots
- Correlation heatmap

Organized into subfolders for easy PDF reporting.
"""

import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")

TARGET_COL = "f_FPro_class"
ID_COLS = ["original_ID"]


def build_binary_target(y: pd.Series) -> pd.Series:
    """Binary target: 1 = non-UPF (0,1,2), 0 = UPF (3)."""
    return y.map(lambda x: 0 if x == 3 else 1)


def infer_feature_types(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """Infer numeric vs categorical feature types."""
    numeric_cols = [
        c for c in df.select_dtypes(include=[np.number]).columns if c != TARGET_COL
    ]
    categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
    return numeric_cols, categorical_cols


def make_output_dirs(base_dir: str) -> Dict[str, str]:
    """Create output directory structure."""
    subfolders = {
        "structure": os.path.join(base_dir, "01_structure"),
        "target": os.path.join(base_dir, "02_target_distribution"),
        "numeric": os.path.join(base_dir, "03_numeric_statistics"),
        "categorical": os.path.join(base_dir, "04_categorical_statistics"),
        "plots": os.path.join(base_dir, "05_plots"),
    }
    for path in subfolders.values():
        os.makedirs(path, exist_ok=True)
    return subfolders


def run_eda(df: pd.DataFrame, outdir: str = "reports") -> Dict[str, object]:
    """Run full EDA and save tables/plots."""
    sns.set(style="whitegrid")
    dirs = make_output_dirs(outdir)
    results: Dict[str, object] = {}

    # ───────────────────────────────────────────────
    # 1. Data Structure
    # ───────────────────────────────────────────────
    numeric_cols, categorical_cols = infer_feature_types(df)
    y_raw = df[TARGET_COL]
    y_bin = build_binary_target(y_raw)
    df_tmp = df.copy()
    df_tmp["binary_target"] = y_bin

    info_tbl = pd.DataFrame({
        "column": df.columns,
        "dtype": [df[c].dtype for c in df.columns],
        "missing_count": [df[c].isna().sum() for c in df.columns],
        "missing_pct": [df[c].isna().mean() * 100 for c in df.columns],
        "unique_values": [df[c].nunique(dropna=True) for c in df.columns],
        "role": [
            "id" if c in ID_COLS else
            "target" if c == TARGET_COL else
            "numeric" if c in numeric_cols else
            "categorical" if c in categorical_cols else
            "other"
            for c in df.columns
        ],
    })
    info_tbl.to_csv(
        os.path.join(dirs["structure"], "columns_overview.csv"),
        index=False,
    )
    results["structure"] = info_tbl

    # ───────────────────────────────────────────────
    # 2. Target Distribution
    # ───────────────────────────────────────────────
    raw_counts = y_raw.value_counts().sort_index()
    raw_counts.to_csv(
        os.path.join(dirs["target"], "target_raw_counts.csv"), header=["count"]
    )
    plt.figure(figsize=(6, 4))
    sns.barplot(x=raw_counts.index.astype(str), y=raw_counts.values)
    plt.title("Raw Target Distribution (f_FPro_class)")
    plt.xlabel("Class (0–3)")
    plt_ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(dirs["target"], "target_raw_distribution.png"))
    plt.close()

    bin_counts = y_bin.value_counts().sort_index()
    bin_counts.to_csv(
        os.path.join(dirs["target"], "target_binary_counts.csv"), header=["count"]
    )
    plt.figure(figsize=(6, 4))
    sns.barplot(x=bin_counts.index.astype(str), y=bin_counts.values)
    plt.title("Binary Target Distribution (1 = non-UPF, 0 = UPF)")
    plt.xlabel("Binary class")
    plt.ylabel("Count")
    plt.tight.tight_layout()
    plt.savefig(os.path.join(dirs["target"], "target_binary_distribution.png"))
    plt.close()

    # ───────────────────────────────────────────────
    # 3. Numeric Descriptive Statistics
    # ───────────────────────────────────────────────
    if numeric_cols:
        desc_stats = df[numeric_cols].describe().T
        desc_stats.to_csv(
            os.path.join(dirs["numeric"], "numeric_describe_overall.csv")
        )
        results["numeric_describe_overall"] = desc_stats

        by_class = df_tmp.groupby("binary_target")[numeric_cols].describe().T
        by_class.to_csv(
            os.path.join(dirs["numeric"], "numeric_describe_by_class.csv")
        )
        results["numeric_describe_by_class"] = by_class

        # Histograms
        for col in numeric_cols:
            plt.figure(figsize=(6, 4))
            sns.histplot(df[col], bins=30)
            plt.title(f"Histogram: {col}")
            plt.tight_layout()
            plt.savefig(os.path.join(
                dirs["plots"],
                f"hist_{col.replace(' ', '_').replace('/', '_')}.png"))
            plt.close()

        # Boxplots
        for col in numeric_cols:
            plt.figure(figsize=(6, 4))
            sns.boxplot(x=df_tmp["binary_target"], y=df_tmp[col])
            plt.title(f"Boxplot by class: {col}")
            plt.xlabel("Binary target (0 = UPF, 1 = non-UPF)")
            plt.tight_layout()
            plt.savefig(os.path.join(
                dirs["plots"], 
                f"box_{col.replace(' ', '_').replace('/', '_')}.png"))
            plt.close()

        # Correlation heatmap
        corr = df[numeric_cols + [TARGET_COL]].corr()
        corr.to_csv(os.path.join(dirs["numeric"], "correlation_matrix.csv"))
        results["correlation"] = corr

        plt.figure(figsize=(12, 8))
        sns.heatmap(corr, annot=False, cmap="coolwarm", center=0)
        plt.title("Correlation Heatmap")
        plt.tight_layout()
        plt.savefig(os.path.join(dirs["plots"], "corr_heatmap.png"))
        plt.close()

    # ───────────────────────────────────────────────
    # 4. Categorical Summaries
    # ───────────────────────────────────────────────
    cat_summary = {}
    for col in categorical_cols:
        vc = df[col].value_counts(dropna=False)

        vc.to_csv(os.path.join(
            dirs["categorical"],
            f"{col.replace(' ', '_')}_value_counts_full.csv"
        ))

        top20 = vc.head(20)
        top20.to_csv(os.path.join(
            dirs["categorical"],
            f"{col.replace(' ', '_')}_top20.csv"
        ))

        plt.figure(figsize=(8, 5))
        sns.barplot(x=top20.values, y=top20.index.astype(str))
        plt.title(f"Top 20: {col}")
        plt.tight_layout()
        plt.savefig(os.path.join(
            dirs["plots"],
            f"bar_top20_{col.replace(' ', '_')}.png"))
        plt.close()

        cat_summary[col] = vc

    results["categorical"] = cat_summary

    print(f"[EDA] Completed. All outputs saved to '{outdir}'.")
    return results
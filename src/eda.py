# src/eda.py
"""
Professional EDA Module
Ultra-Processed Food Classification Project

Includes:
- Data structure summary
- Raw + binary target distribution
- Numeric stats (overall + by class)
- Missing-value heatmap (percentage per feature)
- Correlation heatmap
- Clean histograms (auto skew fix + smart bins)
- Box/violin plots by class (no clipping)
- Scaled pairplot (sampled)
- Robust bivariate plots (smart hexbin / log1p scatter)
- Categorical top-frequency plots
- Z-score outlier summary
- Cramér's V for categorical correlation
"""

import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from scipy.stats import zscore, chi2_contingency
from sklearn.preprocessing import StandardScaler

sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 300

TARGET_COL = "f_FPro_class"

RAW_CATEGORICAL = ["name", "store", "food category", "brand"]

# Assignment-provided numeric feature candidates
NUMERIC_CANDIDATES = [
    "price", "price percal", "package_weight",
    "Protein", "Total Fat", "Carbohydrate", "Sugars, total",
    "Fiber, total dietary", "Calcium", "Iron", "Sodium",
    "Cholesterol", "Fatty acids, total saturated",
]

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def binary_map(series: pd.Series) -> pd.Series:
    """Binary target: 1 = non-UPF (0,1,2), 0 = UPF (3)."""
    return series.map(lambda x: 0 if x == 3 else 1)

def smart_bins(series: pd.Series) -> int:
    """Freedman–Diaconis rule for optimal histogram bin width."""
    series = series.dropna()
    if len(series) < 10:
        return 10
    iqr = series.quantile(0.75) - series.quantile(0.25)
    if iqr == 0:
        return 30
    bin_width = 2 * iqr / (len(series) ** (1 / 3))
    bins = int((series.max() - series.min()) / bin_width)
    return max(10, min(100, bins))

def cramers_v(col1: pd.Series, col2: pd.Series) -> float:
    """Cramér's V for categorical association."""
    confusion = pd.crosstab(col1, col2)
    chi2 = chi2_contingency(confusion)[0]
    n = confusion.sum().sum()
    phi2 = chi2 / n
    r, k = confusion.shape
    return np.sqrt(phi2 / max(1, min(k - 1, r - 1)))

def safe_bivariate_plot(df: pd.DataFrame, x: str, y: str, outdir: str) -> None:
    """
    Robust bivariate plot for numeric pairs:
    - If ranges are comparable → hexbin
    - If extremely skewed → log1p scatter colored by binary_target

    Avoids broken hexbins & unstable KDE.
    """
    if x not in df.columns or y not in df.columns:
        return

    tmp = df[[x, y, "binary_target"]].dropna()
    if tmp.empty:
        return

    x_vals = tmp[x]
    y_vals = tmp[y]

    x_range = x_vals.max() - x_vals.min()
    y_range = y_vals.max() - y_vals.min()
    min_range = max(min(x_range, y_range), 1e-9)
    ratio = max(x_range, y_range) / min_range

    plt.figure(figsize=(6, 5))

    # If extremely skewed → log1p scatter
    if ratio > 25 or x_range > 5000 or y_range > 5000:
        tmp["x_log"] = np.log1p(tmp[x])
        tmp["y_log"] = np.log1p(tmp[y])
        sns.scatterplot(
            data=tmp,
            x="x_log",
            y="y_log",
            hue="binary_target",
            alpha=0.35,
            edgecolor=None,
        )
        plt.xlabel(f"log1p({x})")
        plt.ylabel(f"log1p({y})")
        plt.title(f"Scatter (log1p): {x} vs {y}")
        fname = f"bivar_scatter_log1p_{x}_{y}.png"
    else:
        hb = plt.hexbin(x_vals, y_vals, gridsize=40, cmap="viridis")
        plt.xlabel(x)
        plt.ylabel(y)
        plt.title(f"Hexbin: {x} vs {y}")
        cb = plt.colorbar(hb)
        cb.set_label("count")
        fname = f"bivar_hexbin_{x}_{y}.png"

    plt.tight_layout()
    plt.savefig(os.path.join(outdir, fname))
    plt.close()

def infer_feature_types(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """
    Infer numeric and categorical feature columns from the dataframe.
    - Numeric: all numeric columns except the target.
    - Categorical: all object (string) columns.
    """
    numeric_cols = [
        c for c in df.select_dtypes(include=[np.number]).columns
        if c != TARGET_COL
    ]
    categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
    return numeric_cols, categorical_cols

def make_output_dirs(base_dir: str) -> Dict[str, str]:
    """
    Create a small directory structure under base_dir for nicer organization.
    Returns a dict with logical names -> folder paths.
    """
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

# ------------------------------------------------------------
# Main EDA
# ------------------------------------------------------------
def run_eda(df: pd.DataFrame, outdir: str = "reports") -> dict:
    """
    Run full advanced EDA and write all outputs under `outdir`
    using the 01/02/03/04/05 subfolder structure.
    """
    os.makedirs(outdir, exist_ok=True)
    subdirs = make_output_dirs(outdir)

    print(f"[EDA] Running advanced EDA → {outdir}")
    results: dict = {}

    # Validate target
    if TARGET_COL not in df.columns:
        raise ValueError(f"Dataset is missing required target column: {TARGET_COL}")

    # Auto-detect types, but intersect with assignment numeric list
    auto_numeric, auto_categorical = infer_feature_types(df)
    numeric_cols = [c for c in auto_numeric if c in NUMERIC_CANDIDATES]
    categorical_cols = [c for c in auto_categorical if c in RAW_CATEGORICAL]

    results["numeric_cols"] = numeric_cols
    results["categorical_cols"] = categorical_cols

    # Targets
    y_raw = df[TARGET_COL]
    y_bin = binary_map(y_raw)
    df_tmp = df.copy()
    df_tmp["binary_target"] = y_bin

    # ============================================================
    # 1. Data structure + missingness
    # ============================================================
    info_tbl = pd.DataFrame({
        "column": df.columns,
        "dtype": df.dtypes.astype(str),
        "missing_count": df.isna().sum(),
        "missing_pct": df.isna().mean() * 100,
        "unique_values": df.nunique(dropna=True),
    })
    info_tbl.to_csv(os.path.join(subdirs["structure"], "structure_columns.csv"), index=False)
    results["structure"] = info_tbl

    # Missing-value heatmap (percentage per feature)
    missing_pct = df.isna().mean().to_frame("missing_pct")

    plt.figure(figsize=(14, 3))
    ax = sns.heatmap(
        missing_pct.T,
        annot=True,
        fmt=".2%",
        cmap="YlOrRd",
        cbar=False,
        linewidths=1,
        linecolor="black",
        annot_kws={"fontsize": 10, "color": "black"},
    )
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    plt.title("Missing Value Percentage per Feature", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(subdirs["structure"], "missing_value_heatmap.png"))
    plt.close()

    # ============================================================
    # 2. Target distribution
    # ============================================================
    raw_counts = y_raw.value_counts().sort_index()
    raw_counts.to_csv(os.path.join(subdirs["target"], "target_raw_distribution.csv"))
    plt.figure(figsize=(6, 4))
    sns.barplot(x=raw_counts.index.astype(str), y=raw_counts.values)
    plt.title("Raw Target Distribution (f_FPro_class)")
    plt.xlabel("Class (0–3)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(subdirs["target"], "target_raw_distribution.png"))
    plt.close()

    bin_counts = y_bin.value_counts().sort_index()
    bin_counts.to_csv(os.path.join(subdirs["target"], "target_binary_distribution.csv"))
    plt.figure(figsize=(6, 4))
    sns.barplot(x=bin_counts.index.astype(str), y=bin_counts.values)
    plt.title("Binary Target Distribution (1 = non-UPF, 0 = UPF)")
    plt.xlabel("Binary Class")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(subdirs["target"], "target_binary_distribution.png"))
    plt.close()

    # ============================================================
    # 3. Numeric descriptive statistics + correlation
    # ============================================================
    if numeric_cols:
        desc_overall = df[numeric_cols].describe().T
        desc_overall.to_csv(
            os.path.join(subdirs["numeric"], "descriptive_stats_overall.csv")
        )
        results["descriptive_overall"] = desc_overall

        by_class = df_tmp.groupby("binary_target")[numeric_cols].describe().T
        by_class.to_csv(
            os.path.join(subdirs["numeric"], "descriptive_stats_by_class.csv")
        )
        results["descriptive_by_class"] = by_class

        # Correlation heatmap
        corr = df[numeric_cols].corr()
        plt.figure(figsize=(12, 9))
        sns.heatmap(corr, cmap="coolwarm", center=0, annot=False)
        plt.title("Correlation Heatmap (Numeric Features)")
        plt.tight_layout()
        plt.savefig(os.path.join(subdirs["numeric"], "correlation_heatmap.png"))
        plt.close()
        results["correlation_matrix"] = corr

    # ============================================================
    # 4. Histograms (with optional log) + box/violin by class
    # ============================================================
    plots_dir = subdirs["plots"]

    for col in numeric_cols:
        series = df[col].dropna()
        if series.empty:
            continue

        skew_val = series.skew()
        use_log = skew_val > 1.5 and (series > 0).sum() > 10
        log_series = np.log1p(series) if use_log else None

        # Regular histogram
        plt.figure(figsize=(6, 4))
        sns.histplot(series, bins=smart_bins(series), kde=True)
        plt.title(f"Histogram: {col}")
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f"hist_{col.replace(' ', '_')}.png"))
        plt.close()

        # Log histogram if highly skewed
        if log_series is not None:
            plt.figure(figsize=(6, 4))
            sns.histplot(log_series, bins=smart_bins(log_series), kde=True)
            plt.title(f"Log Histogram: {col}")
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, f"loghist_{col.replace(' ', '_')}.png"))
            plt.close()

        # Boxplot & violin (NO clipping, per your choice)
        plt.figure(figsize=(6, 4))
        sns.boxplot(x=df_tmp["binary_target"], y=df_tmp[col])
        plt.title(f"Boxplot: {col}")
        plt.xlabel("Binary target (0 = UPF, 1 = non-UPF)")
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f"box_{col.replace(' ', '_')}.png"))
        plt.close()

        plt.figure(figsize=(6, 4))
        sns.violinplot(x=df_tmp["binary_target"], y=df_tmp[col], inner="quartile")
        plt.title(f"Violin: {col}")
        plt.xlabel("Binary target (0 = UPF, 1 = non-UPF)")
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f"violin_{col.replace(' ', '_')}.png"))
        plt.close()

    # ============================================================
    # 5. Pairplot (scaled + sampled)
    # ============================================================
    pp_cols = numeric_cols[:6]  # limit to first 6 numeric features
    if len(pp_cols) >= 2:
        sample_df = df_tmp.sample(min(300, len(df_tmp)), random_state=42)
        try:
            scaler = StandardScaler()
            scaled = scaler.fit_transform(sample_df[pp_cols])
            sample_pp = pd.DataFrame(scaled, columns=pp_cols)
            sample_pp["binary_target"] = sample_df["binary_target"].values

            g = sns.pairplot(
                sample_pp,
                hue="binary_target",
                diag_kind="kde",
                corner=True,
            )
            g.savefig(os.path.join(plots_dir, "pairplot_scaled_sampled.png"))
            plt.close("all")
        except Exception as e:
            print(f"[EDA] Pairplot failed: {e}")

    # ============================================================
    # 6. Robust bivariate plots for selected key pairs
    # ============================================================
    joint_pairs = [
        ("Protein", "Total Fat"),
        ("Carbohydrate", "Sugars, total"),
        ("Sodium", "Cholesterol"),
    ]
    for x, y in joint_pairs:
        try:
            safe_bivariate_plot(df_tmp, x, y, plots_dir)
        except Exception as e:
            print(f"[EDA] Bivariate plot failed for {x} vs {y}: {e}")

    # ============================================================
    # 7. Categorical top-frequencies
    # ============================================================
    for col in categorical_cols:
        vc = df[col].value_counts().head(20)
        if vc.empty:
            continue

        plt.figure(figsize=(10, 5))
        sns.barplot(x=vc.values, y=vc.index, orient="h")
        plt.title(f"Top 20 categories: {col}")
        plt.xlabel("Count")
        plt.ylabel(col)
        plt.tight_layout()
        plt.savefig(os.path.join(subdirs["categorical"], f"top_{col.replace(' ', '_')}.png"))
        plt.close()

    # ============================================================
    # 8. Z-score outlier summary (numeric)
    # ============================================================
    if numeric_cols:
        z_df = df[numeric_cols].apply(zscore)
        outlier_counts = (np.abs(z_df) > 3).sum(axis=1)
        pd.DataFrame({
            "row_index": df.index,
            "n_outlier_features": outlier_counts,
        }).to_csv(os.path.join(subdirs["numeric"], "zscore_outliers.csv"), index=False)

    # ============================================================
    # 9. Cramér’s V categorical correlation
    # ============================================================
    cat_results = []
    for i in range(len(categorical_cols)):
        for j in range(i + 1, len(categorical_cols)):
            c1, c2 = categorical_cols[i], categorical_cols[j]
            v = cramers_v(df[c1].astype(str), df[c2].astype(str))
            cat_results.append({"col1": c1, "col2": c2, "cramers_v": v})

    if cat_results:
        pd.DataFrame(cat_results).to_csv(
            os.path.join(subdirs["categorical"], "cramers_v.csv"), index=False
        )

    print("\n[EDA] Advanced EDA complete.")
    print(f"[EDA] All outputs saved → {outdir}\n")
    return results

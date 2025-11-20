# src/run_pipeline.py
"""
Full pipeline:
 1) EDA
 2) Train/Validation/Test split (stratified, seed = TEAM_ID)
 3) Model training & selection on original data
 4) Outlier detection (KMeans) on training set
 5) Model training & selection after outlier removal
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split

from src.eda import run_eda
from src.preprocess import build_binary_target, TARGET_COL
from src.train_model import (
    detect_outliers_kmeans,
    train_models_for_scenario,
)

# 🔁 Set this to your actual TeamID for reproducibility
TEAM_ID = 12345  # <<< CHANGE THIS TO YOUR TEAM ID


def stratified_train_val_test_split(df: pd.DataFrame, test_size=0.2, val_size=0.2, random_state=42):
    """
    Split df into train, val, test stratified by the binary target.
    val_size is relative to the WHOLE dataset (not just train).
    """
    y_bin = build_binary_target(df[TARGET_COL])

    idx = df.index
    # first split: train+val vs test
    idx_train_val, idx_test = train_test_split(
        idx,
        test_size=test_size,
        stratify=y_bin,
        random_state=random_state,
    )

    y_train_val = y_bin.loc[idx_train_val]
    # second split: train vs val (relative val fraction)
    val_frac_relative = val_size / (1.0 - test_size)
    idx_train, idx_val = train_test_split(
        idx_train_val,
        test_size=val_frac_relative,
        stratify=y_train_val,
        random_state=random_state,
    )

    df_train = df.loc[idx_train].copy()
    df_val = df.loc[idx_val].copy()
    df_test = df.loc[idx_test].copy()

    y_train = y_bin.loc[idx_train].to_numpy()
    y_val = y_bin.loc[idx_val].to_numpy()
    y_test = y_bin.loc[idx_test].to_numpy()

    return df_train, y_train, df_val, y_val, df_test, y_test


def main(input_path: str, outdir: str = "reports"):
    os.makedirs(outdir, exist_ok=True)

    print(f" Loading dataset: {input_path}")
    df = pd.read_csv(input_path)

    print(" Running EDA...")
    run_eda(df, outdir)

    print(" Splitting into train/val/test...")
    df_train, y_train, df_val, y_val, df_test, y_test = stratified_train_val_test_split(
        df, test_size=0.2, val_size=0.2, random_state=TEAM_ID
    )

    # ───────── Scenario 1: Original training set ─────────
    print("\n===== Scenario 1: Original training data =====")
    train_models_for_scenario(
        scenario_name="original",
        df_train=df_train,
        y_train=y_train,
        df_val=df_val,
        y_val=y_val,
        df_test=df_test,
        y_test=y_test,
        reports_dir=outdir,
        model_dir="models",
        random_state=TEAM_ID,
        use_smote=True,
    )

    # ───────── Scenario 2: After outlier removal ─────────
    print("\n===== Scenario 2: After outlier removal (top 5% distances) =====")
    keep_idx = detect_outliers_kmeans(df_train, reports_dir=outdir, top_pct=0.05, random_state=TEAM_ID)
    df_train_clean = df_train.loc[keep_idx].copy()
    y_train_clean = build_binary_target(df_train_clean[TARGET_COL]).to_numpy()

    train_models_for_scenario(
        scenario_name="outlier_removed",
        df_train=df_train_clean,
        y_train=y_train_clean,
        df_val=df_val,
        y_val=y_val,
        df_test=df_test,
        y_test=y_test,
        reports_dir=outdir,
        model_dir="models",
        random_state=TEAM_ID,
        use_smote=True,
    )

    print("\nPipeline complete.")
    print(f"Check metrics in: {outdir}/model_metrics_full.csv")
    print(f"Compare test performance in:")
    print(f"  {outdir}/model_performance_test_original.csv")
    print(f"  {outdir}/model_performance_test_outlier_removed.csv")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input CSV file (train dataset)")
    parser.add_argument("--outdir", default="reports", help="Flat reports folder")
    args = parser.parse_args()

    main(args.input, args.outdir)

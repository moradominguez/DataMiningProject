# src/run_pipeline.py
"""
Full Data Science Pipeline:

 1) EDA
 2) Train/Validation/Test split (stratified, seed = TEAM_ID)
 3) Model training & selection on original data
 4) Outlier detection (KMeans)
 5) Model training after outlier removal

Input CSV must be inside the same directory as this script.
Run with:
    python src/run_pipeline.py train.csv
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split

from eda import run_eda
from preprocess import build_binary_target, TARGET_COL
from train_model import (
    detect_outliers_kmeans,
    train_models_for_scenario,
)

# CHANGE THIS TO YOUR REAL TEAM ID
TEAM_ID = 12345

# ─────────────────────────────────────────────────────────────
# Stratified Train/Val/Test Split
# ─────────────────────────────────────────────────────────────
def stratified_train_val_test_split(df: pd.DataFrame, test_size=0.2, val_size=0.2, random_state=42):
    y_bin = build_binary_target(df[TARGET_COL])
    idx = df.index

    # Train+Val vs Test
    idx_train_val, idx_test = train_test_split(
        idx,
        test_size=test_size,
        stratify=y_bin,
        random_state=random_state,
    )

    y_train_val = y_bin.loc[idx_train_val]

    # Train vs Val
    val_frac_relative = val_size / (1.0 - test_size)
    idx_train, idx_val = train_test_split(
        idx_train_val,
        test_size=val_frac_relative,
        stratify=y_train_val,
        random_state=random_state,
    )

    df_train = df.loc[idx_train].copy()
    df_val   = df.loc[idx_val].copy()
    df_test  = df.loc[idx_test].copy()

    y_train = y_bin.loc[idx_train].to_numpy()
    y_val   = y_bin.loc[idx_val].to_numpy()
    y_test  = y_bin.loc[idx_test].to_numpy()

    return df_train, y_train, df_val, y_val, df_test, y_test

# ─────────────────────────────────────────────────────────────
# Main Pipeline
# ─────────────────────────────────────────────────────────────
def main(input_filename: str, outdir: str = "reports"):
    # Create folders if not exist
    os.makedirs(outdir, exist_ok=True)
    os.makedirs("models", exist_ok=True)

    # CSV is in the same directory as this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(script_dir, input_filename)

    print(f"\n Loading dataset from: {input_path}")
    if not os.path.exists(input_path):
        raise FileNotFoundError(
            f"CSV not found: {input_path}\n"
            f"Place '{input_filename}' in the SAME directory as run_pipeline.py"
        )

    df = pd.read_csv(input_path)

    print("\n Running EDA...")
    run_eda(df, outdir)

    print("\n Splitting into train/val/test...")
    df_train, y_train, df_val, y_val, df_test, y_test = stratified_train_val_test_split(
        df,
        test_size=0.2,
        val_size=0.2,
        random_state=TEAM_ID,
    )

    # ───────── SCENARIO 1: ORIGINAL ─────────
    print("\n===== Scenario 1: Original Training Data =====")
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

    # ───────── SCENARIO 2: OUTLIER REMOVAL ─────────
    print("\n===== Scenario 2: Removing Outliers (top 5%) =====")
    keep_idx = detect_outliers_kmeans(
        df_train,
        reports_dir=outdir,
        top_pct=0.05,
        random_state=TEAM_ID,
    )

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

    print("\n Pipeline Complete!")
    print(f" Full metrics saved to: {outdir}/model_metrics_full.csv")
    print(f"Compare:")
    print(f"- {outdir}/model_performance_test_original.csv")
    print(f"- {outdir}/model_performance_test_outlier_removed.csv")

# ─────────────────────────────────────────────────────────────
# CLI — positional argument
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run the ML pipeline. Use: python src/run_pipeline.py train.csv"
    )
    parser.add_argument(
        "input",
        help="CSV filename located in the same directory as this script",
    )
    parser.add_argument(
        "--outdir",
        default="reports",
        help="Output folder for reports (default: reports)"
    )

    args = parser.parse_args()
    main(args.input, args.outdir)

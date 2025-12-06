# src/train_model.py
"""
Task 3 & 4: Model Training, Selection, and Outlier Experiments

Covers:
- 3. Models: Baseline, Decision Tree, Random Forest, XGBoost
- Model selection via validation set and hyperparameter search
- Metrics: Accuracy, Precision, Recall, F1, AUC for train/val/test
- 4. Outlier Detection:
    * KMeans on nutritional features
    * Elbow method plot
    * Remove top 5% far-from-centroid points
    * Retrain models and compare test performance
"""

import os
import itertools
import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

from imblearn.over_sampling import SMOTE

from src.preprocess import (
    TARGET_COL,
    NUMERIC_CANDIDATES,
    build_binary_target,
    select_columns,
    make_preprocess_pipeline,
)

# ─────────────────────────────
# Baseline classifier (no learning)
# ─────────────────────────────
class BaselineClassifier:
    """Simple baseline classifier (majority or random)."""

    def __init__(self, strategy="majority", random_state=42):
        self.strategy = strategy
        self.majority_class_ = None
        self.random_state = random_state

    def fit(self, X, y):
        counts = np.bincount(y)
        self.majority_class_ = np.argmax(counts)
        np.random.seed(self.random_state)
        return self

    def predict(self, X):
        if self.strategy == "majority":
            return np.full(X.shape[0], self.majority_class_)
        elif self.strategy == "random":
            return np.random.randint(0, 2, size=X.shape[0])
        else:
            raise ValueError("Strategy must be 'majority' or 'random'.")


# ─────────────────────────────
# Metrics helper
# ─────────────────────────────
def compute_metrics(y_true, y_pred, y_prob=None):
    m = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }
    if y_prob is not None:
        try:
            m["roc_auc"] = roc_auc_score(y_true, y_prob)
        except ValueError:
            m["roc_auc"] = np.nan
    else:
        m["roc_auc"] = np.nan
    return m


# ─────────────────────────────
# Train/Val/Test preprocessing
# ─────────────────────────────
def build_preprocessor_and_transform(df_train, df_val, df_test, scale=True):
    """Fit preprocessor on train only, transform train/val/test."""
    numeric_cols, categorical_cols = select_columns(df_train)

    # drop ID + target
    drop_cols = ["original_ID", TARGET_COL]
    X_train_df = df_train.drop(columns=[c for c in drop_cols if c in df_train.columns])
    X_val_df = df_val.drop(columns=[c for c in drop_cols if c in df_val.columns])
    X_test_df = df_test.drop(columns=[c for c in drop_cols if c in df_test.columns])

    pre = make_preprocess_pipeline(numeric_cols, categorical_cols, scale=scale)
    X_train = pre.fit_transform(X_train_df)
    X_val = pre.transform(X_val_df)
    X_test = pre.transform(X_test_df)

    feature_names = pre.get_feature_names_out()
    return pre, feature_names, X_train, X_val, X_test


# ─────────────────────────────
# Outlier detection via KMeans
# ─────────────────────────────
def detect_outliers_kmeans(df_train, reports_dir="reports", top_pct=0.05, random_state=42):
    """
    Use only numeric nutritional features, do elbow plot, then KMeans with a fixed K (e.g., 4),
    rank points by distance to assigned centroid, and return indices of "kept" rows (i.e., not outliers).
    """
    os.makedirs(reports_dir, exist_ok=True)
    numeric_cols = [c for c in NUMERIC_CANDIDATES if c in df_train.columns]

    df_num = df_train[numeric_cols].dropna()
    if df_num.empty:
        print("[Outliers] No numeric data available; skipping outlier detection.")
        return df_train.index  # keep all

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_num.values)

    # Elbow method
    inertias = []
    K_range = range(1, 11)
    for k in K_range:
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        km.fit(X_scaled)
        inertias.append(km.inertia_)

    import matplotlib.pyplot as plt
    plt.figure(figsize=(6, 4))
    plt.plot(list(K_range), inertias, marker="o")
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("Within-cluster sum of squares")
    plt.title("KMeans Elbow Method (training set)")
    plt.tight_layout()

    prefix = f"t{random_state:02d}_"
    plt.savefig(f"{reports_dir}/{prefix}kmeans_elbow_plot.png")
    plt.close()

    # For automation, pick a small reasonable K
    k_best = 4
    km = KMeans(n_clusters=k_best, random_state=random_state, n_init=10)
    km.fit(X_scaled)

    # Distance to closest centroid
    distances = km.transform(X_scaled).min(axis=1)
    n_outliers = max(1, int(len(distances) * top_pct))

    # Largest distances = most "outlier-like"
    outlier_idx_sorted = np.argsort(distances)[-n_outliers:]
    outlier_indices = df_num.index[outlier_idx_sorted]

    keep_index = df_train.index.difference(outlier_indices)
    print(f"[Outliers] Marked {n_outliers} ({top_pct*100:.1f}%) points as outliers.")

    pd.Series(outlier_indices, name="outlier_index").to_csv(
        f"{reports_dir}/{prefix}outliers_indices.csv", index=False
    )

    return keep_index


# ─────────────────────────────
# Main training with hyperparameter search
# ─────────────────────────────
def train_models_for_scenario(
    scenario_name,
    df_train,
    y_train,
    df_val,
    y_val,
    df_test,
    y_test,
    reports_dir="reports",
    model_dir="models",
    random_state=42,
    use_smote=True,
):
    """
    Train all models under a given scenario (e.g., original / outlier_removed)
    with hyperparameter tuning using the validation set.
    Returns a DataFrame with metrics per split (train/val/test).
    """
    from xgboost import XGBClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier

    os.makedirs(reports_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    # Preprocess train/val/test (fit preprocessor on train only)
    _, _, X_train_raw, X_val, X_test = build_preprocessor_and_transform(
        df_train, df_val, df_test, scale=True
    )

    print(
        f"\n[DEBUG] Preprocess shapes:"
        f"\n  X_train_raw: {X_train_raw.shape} | y_train: {len(y_train)}"
        f"\n  X_val:       {X_val.shape} | y_val:   {len(y_val)}"
        f"\n  X_test:      {X_test.shape} | y_test:  {len(y_test)}"
    )

    results = []

    # Define hyperparameter grids (≥ 3 values per key hyperparam)
    model_spaces = {
        "baseline_majority": {
            "model_class": BaselineClassifier,
            "param_grid": {"strategy": ["majority"]},
        },
        "decision_tree": {
            "model_class": DecisionTreeClassifier,
            "param_grid": {
                "max_depth": [3, 5, 7],
                "min_samples_split": [2, 10, 20],
            },
        },
        "random_forest": {
            "model_class": RandomForestClassifier,
            "param_grid": {
                "n_estimators": [100, 200, 300],
                "max_depth": [5, 10, None],
            },
        },
        "xgboost": {
            "model_class": XGBClassifier,
            "param_grid": {
                "n_estimators": [200, 300, 400],
                "max_depth": [3, 5, 7],
                "learning_rate": [0.05, 0.1, 0.2],
            },
        },
    }

    for model_name, cfg in model_spaces.items():
        ModelClass = cfg["model_class"]
        param_grid = cfg["param_grid"]

        print(f"\n[Scenario: {scenario_name}] Tuning model: {model_name}")

        # Baseline must NOT use SMOTE
        if model_name == "baseline_majority":
            model = ModelClass(strategy="majority", random_state=random_state)

            model.fit(X_train_raw, y_train)
            y_train_pred = model.predict(X_train_raw)
            y_val_pred = model.predict(X_val)
            y_test_pred = model.predict(X_test)

            print(f"[DEBUG] Baseline lengths:\n  y_train: {len(y_train)} | y_train_pred: {len(y_train_pred)}")

            m_train = compute_metrics(y_train, y_train_pred)
            m_val = compute_metrics(y_val, y_val_pred)
            m_test = compute_metrics(y_test, y_test_pred)

            for split, metrics in [("train", m_train), ("val", m_val), ("test", m_test)]:
                row = {"scenario": scenario_name, "model": model_name, "split": split, "params": "{}"}
                row.update(metrics)
                results.append(row)

            model_path = os.path.join(model_dir, f"{model_name}_{scenario_name}.pkl")
            joblib.dump(model, model_path)
            print(f"[{model_name}] Saved → {model_path}")
            continue

        # SMOTE only on train for non-baseline models
        if use_smote:
            smote = SMOTE(random_state=random_state)
            X_train, y_train_bal = smote.fit_resample(X_train_raw, y_train)
        else:
            X_train, y_train_bal = X_train_raw, y_train

        print(f"[DEBUG] After SMOTE (non-baseline only):\n  X_train: {X_train.shape} | y_train_bal: {len(y_train_bal)}")

        # generate all combos
        param_keys = list(param_grid.keys())
        all_param_combos = list(itertools.product(*[param_grid[k] for k in param_keys]))

        # ✅ Speed: sample a subset for xgboost while still defining >=3 values per hyperparam
        if model_name == "xgboost":
            rng = np.random.RandomState(random_state)
            if len(all_param_combos) > 10:
                pick = rng.choice(len(all_param_combos), size=10, replace=False)
                all_param_combos = [all_param_combos[j] for j in pick]

        best_f1_val = -1.0
        best_model = None
        best_params = None

        for i, combo in enumerate(all_param_combos, start=1):
            params = dict(zip(param_keys, combo))

            if model_name == "xgboost":
                print(f"[xgboost] Fit {i}/{len(all_param_combos)} params={params}")
                model = ModelClass(
                    **params,
                    eval_metric="logloss",
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=random_state,
                    n_jobs=-1,
                    tree_method="hist",
                )
            else:
                model = ModelClass(**params, random_state=random_state)

            model.fit(X_train, y_train_bal)

            y_val_pred = model.predict(X_val)
            f1_val = f1_score(y_val, y_val_pred, zero_division=0)

            if f1_val > best_f1_val:
                best_f1_val = f1_val
                best_model = model
                best_params = params

        # evaluate best model on ORIGINAL train/val/test
        y_train_pred = best_model.predict(X_train_raw)
        y_val_pred = best_model.predict(X_val)
        y_test_pred = best_model.predict(X_test)

        y_train_prob = best_model.predict_proba(X_train_raw)[:, 1] if hasattr(best_model, "predict_proba") else None
        y_val_prob = best_model.predict_proba(X_val)[:, 1] if hasattr(best_model, "predict_proba") else None
        y_test_prob = best_model.predict_proba(X_test)[:, 1] if hasattr(best_model, "predict_proba") else None

        m_train = compute_metrics(y_train, y_train_pred, y_train_prob)
        m_val = compute_metrics(y_val, y_val_pred, y_val_prob)
        m_test = compute_metrics(y_test, y_test_pred, y_test_prob)

        for split, metrics in [("train", m_train), ("val", m_val), ("test", m_test)]:
            row = {"scenario": scenario_name, "model": model_name, "split": split, "params": str(best_params)}
            row.update(metrics)
            results.append(row)

        model_path = os.path.join(model_dir, f"{model_name}_{scenario_name}.pkl")
        joblib.dump(best_model, model_path)
        print(f"[{model_name}] Best params: {best_params}")
        print(f"[{model_name}] Saved → {model_path}")

    df_results = pd.DataFrame(results)

    prefix = f"t{random_state:02d}_"
    metrics_path = f"{reports_dir}/{prefix}model_metrics_full.csv"

    # ✅ Append across scenarios (original + outlier_removed),
    # and avoid duplicates by deleting the CSV before a fresh run.
    if os.path.exists(metrics_path):
        df_existing = pd.read_csv(metrics_path)
        df_all = pd.concat([df_existing, df_results], ignore_index=True)
    else:
        df_all = df_results
    df_all.to_csv(metrics_path, index=False)

    test_summary = df_results[df_results["split"] == "test"]
    test_summary.to_csv(
        f"{reports_dir}/{prefix}model_performance_test_{scenario_name}.csv", index=False
    )

    return df_results

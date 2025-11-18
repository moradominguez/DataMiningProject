# src/train_model.py
"""
Task 3: Model Training for Ultra-Processed Food Classifier
----------------------------------------------------------
Models:
(1) Baseline (majority/random)
(2) Decision Tree
(3) Random Forest
(4) XGBoost (complex model)

All models output:
  - Metrics (accuracy, precision, recall, F1, ROC-AUC)
  - Saved model file (.pkl)
  - Report CSV summary
"""

import os
import joblib
import numpy as np
import pandas as pd

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score
)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier


# ─────────────────────────────
# Utility: Baseline Classifier
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
# Utility: Metrics
# ─────────────────────────────
def compute_metrics(y_true, y_pred, y_prob=None):
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }
    if y_prob is not None:
        try:
            metrics["roc_auc"] = roc_auc_score(y_true, y_prob)
        except ValueError:
            metrics["roc_auc"] = np.nan
    return metrics


# ─────────────────────────────
# Core Training Function
# ─────────────────────────────
def train_all_models(X, y, outdir="models", test_size=0.2, random_state=42):
    os.makedirs(outdir, exist_ok=True)
    reports_dir = os.path.join(outdir, "reports")
    os.makedirs(reports_dir, exist_ok=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    models = {
        "baseline_majority": BaselineClassifier("majority"),
        "decision_tree": DecisionTreeClassifier(max_depth=6, random_state=random_state),
        "random_forest": RandomForestClassifier(
            n_estimators=200, max_depth=8, random_state=random_state
        ),
        "xgboost": XGBClassifier(
            n_estimators=300,
            learning_rate=0.1,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=random_state,
            n_jobs=-1,
        ),
    }

    results = []
    for name, model in models.items():
        print(f" Training model: {name}")
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_prob = None
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]

        metrics = compute_metrics(y_test, y_pred, y_prob)
        metrics["model"] = name
        results.append(metrics)

        # save model
        model_path = os.path.join(outdir, f"{name}.pkl")
        joblib.dump(model, model_path)
        print(f" Saved {name} → {model_path}")

    # Save metrics summary
    df_results = pd.DataFrame(results)
    df_results.to_csv(os.path.join(reports_dir, "model_performance.csv"), index=False)
    print(f" Model metrics saved to: {reports_dir}/model_performance.csv")

    return df_results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Preprocessed CSV (from preprocess stage)")
    parser.add_argument("--outdir", default="models", help="Output directory for models and reports")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    X = df.drop(columns=["binary_target"]).values
    y = df["binary_target"].values

    results = train_all_models(X, y, args.outdir)
    print(results)

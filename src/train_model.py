# src/train_model.py
import os
import joblib
import numpy as np
import pandas as pd

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score
)
from sklearn.model_selection import train_test_split


class BaselineClassifier:
    def __init__(self, strategy="majority"):
        self.strategy = strategy

    def fit(self, X, y):
        self.majority = np.argmax(np.bincount(y))
        return self

    def predict(self, X):
        if self.strategy == "majority":
            return np.full(X.shape[0], self.majority)
        return np.random.randint(0, 2, size=X.shape[0])


def compute_metrics(y_true, y_pred, prob=None):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, prob) if prob is not None else np.nan
    }


def train_all_models(X_train, y_train, X_test, y_test, model_dir="models", reports_dir="reports"):
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)

    models = {
        "baseline_majority": BaselineClassifier(),
        "decision_tree": DecisionTreeClassifier(max_depth=6),
        "random_forest": RandomForestClassifier(n_estimators=200, max_depth=8),
        "xgboost": XGBClassifier(
            n_estimators=300, max_depth=5,
            learning_rate=0.1, subsample=0.8,
            colsample_bytree=0.8, eval_metric="logloss"
        )
    }

    results = []

    for name, model in models.items():
        print(f"Training: {name}")
        model.fit(X_train, y_train)

        pred = model.predict(X_test)
        prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

        metrics = compute_metrics(y_test, pred, prob)
        metrics["model"] = name
        results.append(metrics)

        joblib.dump(model, f"{model_dir}/{name}.pkl")
        print(f"Saved → {model_dir}/{name}.pkl")

    df = pd.DataFrame(results)
    df.to_csv(f"{reports_dir}/model_performance.csv", index=False)

    return df

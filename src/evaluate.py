# src/evaluate.py
"""
Task 4: Model Evaluation & Visualization
----------------------------------------
- Loads trained models from /models
- Evaluates them on held-out test set or new data
- Creates:
    * Confusion matrices
    * ROC curves
    * Ranked metric plots
"""

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)


def evaluate_model(model, X_test, y_test, model_name, outdir="reports/evaluation"):
    """Evaluate a single trained model and save plots + metrics."""
    os.makedirs(outdir, exist_ok=True)

    y_pred = model.predict(X_test)
    y_prob = None
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]

    # --- Metrics ---
    metrics = {
        "model": model_name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
    }
    if y_prob is not None:
        try:
            metrics["roc_auc"] = roc_auc_score(y_test, y_prob)
        except Exception:
            metrics["roc_auc"] = np.nan
    else:
        metrics["roc_auc"] = np.nan

    # --- Confusion Matrix ---
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap="Blues")
    plt.title(f"Confusion Matrix: {model_name}")
    plt.tight_layout()
    plt.savefig(f"{outdir}/{model_name}_confusion_matrix.png")
    plt.close()

    # --- ROC Curve ---
    if y_prob is not None:
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        plt.figure()
        plt.plot(fpr, tpr, label=f"{model_name} (AUC={metrics['roc_auc']:.2f})")
        plt.plot([0, 1], [0, 1], "k--", label="Random")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC Curve: {model_name}")
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(f"{outdir}/{model_name}_roc_curve.png")
        plt.close()

    return metrics


def evaluate_all_models(X_test, y_test, model_dir="models", outdir="reports/evaluation"):
    """Evaluate every .pkl model in model_dir."""
    os.makedirs(outdir, exist_ok=True)
    results = []

    for file in os.listdir(model_dir):
        if not file.endswith(".pkl"):
            continue
        path = os.path.join(model_dir, file)
        model_name = os.path.splitext(file)[0]
        print(f"Evaluating model: {model_name}")

        model = joblib.load(path)
        metrics = evaluate_model(model, X_test, y_test, model_name, outdir)
        results.append(metrics)

    df_results = pd.DataFrame(results)
    df_results.to_csv(f"{outdir}/evaluation_summary.csv", index=False)

    # --- Rank Plot ---
    metric_cols = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    long_df = df_results.melt(id_vars="model", value_vars=metric_cols,
                              var_name="metric", value_name="score")

    plt.figure(figsize=(8, 5))
    sns.barplot(data=long_df, x="metric", y="score", hue="model")
    plt.title("Model Performance Comparison")
    plt.legend(title="Model")
    plt.tight_layout()
    plt.savefig(f"{outdir}/model_comparison_barplot.png")
    plt.close()

    print(f"Evaluation complete. Reports saved in {outdir}")
    return df_results

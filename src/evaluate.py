# src/evaluate.py
import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay, roc_curve, roc_auc_score
)

def evaluate_model(model, X_test, y_test, name, outdir="reports"):
    pred = model.predict(X_test)
    prob = None
    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(X_test)[:, 1]

    # Confusion Matrix
    cm = confusion_matrix(y_test, pred)
    ConfusionMatrixDisplay(confusion_matrix=cm).plot(cmap="Blues")
    plt.title(f"Confusion Matrix - {name}")
    plt.tight_layout()
    plt.savefig(f"{outdir}/{name}_confusion_matrix.png")
    plt.close()

    # ROC Curve
    if prob is not None:
        fpr, tpr, _ = roc_curve(y_test, prob)
        auc = roc_auc_score(y_test, prob)

        plt.figure()
        plt.plot(fpr, tpr, label=f"AUC={auc:.2f}")
        plt.plot([0, 1], [0, 1], "k--")
        plt.title(f"ROC Curve - {name}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{outdir}/{name}_roc_curve.png")
        plt.close()

    return {
        "model": name,
        "accuracy": (pred == y_test).mean(),
        "roc_auc": roc_auc_score(y_test, prob) if prob is not None else None
    }


def evaluate_all_models(X_test, y_test, model_dir="models", outdir="reports"):
    os.makedirs(outdir, exist_ok=True)
    results = []

    for file in os.listdir(model_dir):
        if not file.endswith(".pkl"):
            continue

        name = file[:-4]
        model = joblib.load(f"{model_dir}/{file}")
        print(f"Evaluating: {name}")

        metrics = evaluate_model(model, X_test, y_test, name, outdir)
        results.append(metrics)

    df = pd.DataFrame(results)
    df.to_csv(f"{outdir}/evaluation_summary.csv", index=False)

    # Barplot comparison
    plt.figure(figsize=(7, 5))
    sns.barplot(data=df, x="model", y="accuracy")
    plt.title("Model Accuracy Comparison")
    plt.tight_layout()
    plt.savefig(f"{outdir}/model_comparison_barplot.png")
    plt.close()

    return df

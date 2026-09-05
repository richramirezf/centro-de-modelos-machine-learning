import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from src_telco.data_loader import load_telco_data
from src_telco.quality import run_quality_pipeline

TEST_SIZE = 0.2
RANDOM_STATE = 42
CLASS_LABELS = ["No Churn (0)", "Churn (1)"]

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"


def prepare_data() -> pd.DataFrame:
    df = load_telco_data(DATA_PATH)
    df = run_quality_pipeline(df)
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
    return df


def build_confusion(X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    pipeline = joblib.load(MODELS_DIR / "churn_model.joblib")
    y_train_pred = pipeline.predict(X_train)
    y_test_pred = pipeline.predict(X_test)
    y_test_proba = pipeline.predict_proba(X_test)[:, 1]

    tn, fp, fn, tp = confusion_matrix(y_train, y_train_pred, labels=[0, 1]).ravel()

    return {
        "model_name": "churn_model",
        "class_labels": CLASS_LABELS,
        "matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
        "accuracy": float(accuracy_score(y_train, y_train_pred)),
        "precision_class_1": float(precision_score(y_train, y_train_pred)),
        "recall_class_1": float(recall_score(y_train, y_train_pred)),
        "samples": int(len(y_train)),
        "test": {
            "accuracy": float(accuracy_score(y_test, y_test_pred)),
            "roc_auc": float(roc_auc_score(y_test, y_test_proba)),
            "precision_churn": float(precision_score(y_test, y_test_pred)),
            "recall_churn": float(recall_score(y_test, y_test_pred)),
        },
    }


def plot_confusion(report: dict, output_path: Path) -> None:
    matrix = report["matrix"]
    fig, ax = plt.subplots(figsize=(4.6, 4))
    im = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks([0, 1], CLASS_LABELS)
    ax.set_yticks([0, 1], CLASS_LABELS)
    ax.set_xlabel("Predicción")
    ax.set_ylabel("Real")
    ax.set_title(f"Matriz de confusión (entrenamiento)\nXGBoost Churn — {report['samples']} muestras")

    for i in range(2):
        for j in range(2):
            ax.text(j, i, matrix[i][j], ha="center", va="center", color="white" if matrix[i][j] > 40 else "#0b3d2e", fontsize=14)

    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main() -> None:
    df = prepare_data()
    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )

    report = build_confusion(X_train, y_train, X_test, y_test)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    plot_confusion(report, REPORTS_DIR / "confusion_churn_model.png")

    (MODELS_DIR / "confusion_report_telco.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )

    print(f"Train: TN={report['TN']} FP={report['FP']} FN={report['FN']} TP={report['TP']} | acc={report['accuracy']:.4f} prec(churn)={report['precision_class_1']:.4f} recall(churn)={report['recall_class_1']:.4f}")
    print(f"Test: acc={report['test']['accuracy']:.4f} auc={report['test']['roc_auc']:.4f} prec(churn)={report['test']['precision_churn']:.4f} recall(churn)={report['test']['recall_churn']:.4f}")


if __name__ == "__main__":
    main()

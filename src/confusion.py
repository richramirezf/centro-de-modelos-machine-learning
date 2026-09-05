import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score
from sklearn.model_selection import train_test_split

from src.data_loader import load_german_credit_data
from src.quality import encode_risk, impute_account_missing

TEST_SIZE = 0.2
RANDOM_STATE = 42
CLASS_LABELS = ["good (0)", "bad (1)"]

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "german_credit_data.csv"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"


def prepare_data() -> pd.DataFrame:
    df = load_german_credit_data(DATA_PATH)
    df = impute_account_missing(df)
    return encode_risk(df)


def build_confusion(model_name: str, X_train: pd.DataFrame, y_train: pd.Series) -> dict:
    pipeline = joblib.load(MODELS_DIR / f"{model_name}.joblib")
    y_pred = pipeline.predict(X_train)

    tn, fp, fn, tp = confusion_matrix(y_train, y_pred, labels=[0, 1]).ravel()

    return {
        "model_name": model_name,
        "matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
        "accuracy": float(accuracy_score(y_train, y_pred)),
        "precision_class_1": float(precision_score(y_train, y_pred)),
        "recall_class_1": float(recall_score(y_train, y_pred)),
        "samples": int(len(y_train)),
    }


def plot_confusion(report: dict, output_path: Path) -> None:
    matrix = report["matrix"]
    fig, ax = plt.subplots(figsize=(4.6, 4))
    im = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks([0, 1], CLASS_LABELS)
    ax.set_yticks([0, 1], CLASS_LABELS)
    ax.set_xlabel("Predicción")
    ax.set_ylabel("Real")
    ax.set_title(f"Matriz de confusión (entrenamiento)\n{report['model_name']} — {report['samples']} muestras")

    for i in range(2):
        for j in range(2):
            ax.text(j, i, matrix[i][j], ha="center", va="center", color="white" if matrix[i][j] > 40 else "#0b3d2e", fontsize=14)

    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main() -> None:
    df = prepare_data()
    X = df.drop(columns=["Risk_num"])
    y = df["Risk_num"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    records = []
    for model_name in ["logistic_model", "xgb_model"]:
        report = build_confusion(model_name, X_train, y_train)
        records.append(report)
        plot_confusion(report, REPORTS_DIR / f"confusion_{model_name}.png")

    (MODELS_DIR / "confusion_report.json").write_text(
        json.dumps({"target": "Risk_num (1 = bad / default)", "classes": CLASS_LABELS, "models": records}, indent=2),
        encoding="utf-8",
    )

    for r in records:
        print(f"{r['model_name']}: TN={r['TN']} FP={r['FP']} FN={r['FN']} TP={r['TP']} | acc={r['accuracy']:.4f} prec(bad)={r['precision_class_1']:.4f} recall(bad)={r['recall_class_1']:.4f}")


if __name__ == "__main__":
    main()
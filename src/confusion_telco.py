import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

from src.confusion_common import confusion_metrics, plot_confusion
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


def build_confusion(X_train, y_train, X_test, y_test) -> dict:
    pipeline = joblib.load(MODELS_DIR / "churn_model.joblib")
    y_train_pred = pipeline.predict(X_train)
    y_test_pred = pipeline.predict(X_test)
    y_test_proba = pipeline.predict_proba(X_test)[:, 1]

    report = confusion_metrics(y_train, y_train_pred, labels=[0, 1])
    report["model_name"] = "churn_model"
    report["class_labels"] = CLASS_LABELS
    report["test"] = {
        "accuracy": float(accuracy_score(y_test, y_test_pred)),
        "roc_auc": float(roc_auc_score(y_test, y_test_proba)),
        "precision_churn": float(precision_score(y_test, y_test_pred, zero_division=0)),
        "recall_churn": float(recall_score(y_test, y_test_pred, zero_division=0)),
    }
    return report


def main() -> None:
    df = prepare_data()
    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )

    report = build_confusion(X_train, y_train, X_test, y_test)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    plot_confusion(
        report,
        REPORTS_DIR / "confusion_churn_model.png",
        labels=CLASS_LABELS,
        title="XGBoost Churn",
    )

    (MODELS_DIR / "confusion_report_telco.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )

    print(f"Train: TN={report['TN']} FP={report['FP']} FN={report['FN']} TP={report['TP']} | acc={report['accuracy']:.4f} prec(churn)={report['precision_class_1']:.4f} recall(churn)={report['recall_class_1']:.4f}")
    print(f"Test: acc={report['test']['accuracy']:.4f} auc={report['test']['roc_auc']:.4f} prec(churn)={report['test']['precision_churn']:.4f} recall(churn)={report['test']['recall_churn']:.4f}")


if __name__ == "__main__":
    main()

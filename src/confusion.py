import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

from src.confusion_common import confusion_metrics, plot_confusion
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


def build_confusion(model_name: str, X_train, y_train) -> dict:
    pipeline = joblib.load(MODELS_DIR / f"{model_name}.joblib")
    y_pred = pipeline.predict(X_train)

    report = confusion_metrics(y_train, y_pred, labels=[0, 1])
    report["model_name"] = model_name
    return report


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
        plot_confusion(
            report,
            REPORTS_DIR / f"confusion_{model_name}.png",
            labels=CLASS_LABELS,
            title=model_name,
        )

    (MODELS_DIR / "confusion_report.json").write_text(
        json.dumps({"target": "Risk_num (1 = bad / default)", "classes": CLASS_LABELS, "models": records}, indent=2),
        encoding="utf-8",
    )

    for r in records:
        print(f"{r['model_name']}: TN={r['TN']} FP={r['FP']} FN={r['FN']} TP={r['TP']} | acc={r['accuracy']:.4f} prec(bad)={r['precision_class_1']:.4f} recall(bad)={r['recall_class_1']:.4f}")


if __name__ == "__main__":
    main()

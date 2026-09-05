import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from src_telco.data_loader import load_telco_data
from src_telco.quality import run_quality_pipeline


NUMERIC_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges"]
CATEGORICAL_FEATURES = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]

TEST_SIZE = 0.2
RANDOM_STATE = 42

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
MODELS_DIR = PROJECT_ROOT / "models"


def build_pipeline() -> Pipeline:
    """Return a sklearn Pipeline with preprocessing + XGBClassifier."""
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                XGBClassifier(
                    n_estimators=200,
                    max_depth=5,
                    learning_rate=0.1,
                    eval_metric="logloss",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )
    return pipeline


def train(csv_path: str | Path | None = None) -> dict:
    """Full training routine: load → clean → train → evaluate → save."""
    df = load_telco_data(csv_path)
    df = run_quality_pipeline(df)
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    print(f"[train] Train={len(X_train):,}  Test={len(X_test):,}")

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    metrics = {
        "test_accuracy": float(accuracy_score(y_test, y_pred)),
        "test_roc_auc": float(roc_auc_score(y_test, y_proba)),
        "test_precision_churn": float(precision_score(y_test, y_pred)),
        "test_recall_churn": float(recall_score(y_test, y_pred)),
    }
    for key, value in metrics.items():
        print(f"[train] {key}: {value:.4f}")

    MODELS_DIR.mkdir(exist_ok=True)
    model_path = MODELS_DIR / "churn_model.joblib"
    joblib.dump(pipeline, model_path)
    print(f"[train] Model saved → {model_path}")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the Telco churn model.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH, help="Path to the Telco CSV")
    args = parser.parse_args()
    train(args.data)

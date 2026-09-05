import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from src.data_loader import load_german_credit_data
from src.quality import encode_risk, impute_account_missing

CATEGORICAL_COLUMNS = ["Sex", "Housing", "Saving accounts", "Checking account", "Purpose"]
NUMERIC_COLUMNS = ["Age", "Credit amount", "Duration"]
TARGET = "Risk_num"
TEST_SIZE = 0.2
RANDOM_STATE = 42

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "german_credit_data.csv"
MODELS_DIR = PROJECT_ROOT / "models"


def prepare_data(path: str | Path) -> pd.DataFrame:
    """Load the dataset and apply the quality pipeline from phase 1."""
    df = load_german_credit_data(path)
    df = impute_account_missing(df)
    df = encode_risk(df)
    return df


def build_preprocessor() -> ColumnTransformer:
    """One-hot encode categoricals and standardize numerics."""
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_COLUMNS),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_COLUMNS),
        ]
    )


def build_pipeline(model) -> Pipeline:
    return Pipeline([("preprocessor", build_preprocessor()), ("classifier", model)])


def print_report(results: dict) -> None:
    header = f"{'Model':<22}{'ROC-AUC':>12}{'Recall (class 1)':>18}"
    print("=" * len(header))
    print("Comparative evaluation report")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for name, metrics in results.items():
        print(f"{name:<22}{metrics['ROC-AUC']:>12.4f}{metrics['Recall (class 1)']:>18.4f}")
    print("=" * len(header))


def train_and_evaluate(path: str | Path = DEFAULT_DATA_PATH) -> dict:
    df = prepare_data(path)
    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )

    models = {
        "Logistic Regression": build_pipeline(
            LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
        ),
        "XGBoost": build_pipeline(
            XGBClassifier(eval_metric="logloss", random_state=RANDOM_STATE)
        ),
    }

    results = {}
    fitted = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        results[name] = {
            "ROC-AUC": roc_auc_score(y_test, y_proba),
            "Recall (class 1)": recall_score(y_test, y_pred),
        }
        fitted[name] = model

    print_report(results)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(fitted["Logistic Regression"], MODELS_DIR / "logistic_model.joblib")
    joblib.dump(fitted["XGBoost"], MODELS_DIR / "xgb_model.joblib")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and compare credit scoring models.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH, help="Path to german_credit_data.csv")
    args = parser.parse_args()
    train_and_evaluate(args.data)


if __name__ == "__main__":
    main()
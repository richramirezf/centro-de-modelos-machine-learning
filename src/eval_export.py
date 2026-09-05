"""Exporta arrays de evaluación (test) por modelo de clasificación para los
laboratorios interactivos del portal (matriz dinámica, curva ROC).

El runtime de Streamlit NO accede a los CSV: estos .npz se generan offline.
"""

import sys

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

import joblib

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
EVAL_DIR = MODELS_DIR / "eval"
TEST_SIZE = 0.2
RANDOM_STATE = 42


def _split(X, y):
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    return X_te, y_te


def _save(name: str, y_test: pd.Series, proba: np.ndarray) -> None:
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        EVAL_DIR / f"{name}.npz",
        y=y_test.to_numpy(dtype=np.int8),
        p=proba.astype(np.float32),
    )
    print(f"[eval] {name}: {len(y_test):,} test samples saved")


def export_credit() -> None:
    csv_path = DATA_DIR / "german_credit_data.csv"
    if not csv_path.exists():
        print("[eval] skip credit: data not found")
        return
    from src.data_loader import load_german_credit_data
    from src.quality import encode_risk, impute_account_missing

    df = load_german_credit_data(csv_path)
    df = impute_account_missing(df)
    df = encode_risk(df)
    X = df.drop(columns=["Risk_num"])
    y = df["Risk_num"]
    X_te, y_te = _split(X, y)

    for label, filename, out in [
        ("Regresión Logística", "logistic_model.joblib", "credit_logreg"),
        ("XGBoost", "xgb_model.joblib", "credit_xgb"),
    ]:
        model = joblib.load(MODELS_DIR / filename)
        p = model.predict_proba(X_te)[:, 1]
        _save(out, y_te, p)
        print(f"[eval]   exported {label}")


def export_churn() -> None:
    csv_path = DATA_DIR / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
    if not csv_path.exists():
        print("[eval] skip churn: data not found")
        return
    from src_telco.data_loader import load_telco_data
    from src_telco.quality import run_quality_pipeline

    df = load_telco_data(csv_path)
    df = run_quality_pipeline(df)
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
    X = df.drop(columns=["Churn"])
    y = df["Churn"]
    X_te, y_te = _split(X, y)

    model = joblib.load(MODELS_DIR / "churn_model.joblib")
    p = model.predict_proba(X_te)[:, 1]
    _save("churn", y_te, p)


def export_noshow() -> None:
    csv_path = DATA_DIR / "KaggleV2-May-2016.csv"
    if not csv_path.exists():
        print("[eval] skip noshow: data not found")
        return
    from src.medical_train import FEATURES, TARGET, build_features, load_noshow_data

    df = build_features(load_noshow_data(csv_path))
    X = df[FEATURES]
    y = df[TARGET]
    X_te, y_te = _split(X, y)

    model = joblib.load(MODELS_DIR / "noshow_model.joblib")
    p = model.predict_proba(X_te)[:, 1]
    _save("noshow", y_te, p)


def main() -> None:
    export_credit()
    export_churn()
    export_noshow()


if __name__ == "__main__":
    main()

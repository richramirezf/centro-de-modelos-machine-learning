import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"

TEST_SIZE = 0.2
RANDOM_STATE = 42

FEATURES = ["metros_cuadrados", "habitaciones", "antiguedad_anios", "tiene_garaje"]
NUMERIC = ["metros_cuadrados", "habitaciones", "antiguedad_anios"]
BINARY = ["tiene_garaje"]
TARGET = "precio_usd"


def simulate_housing(n: int = 15000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "metros_cuadrados": rng.integers(40, 301, size=n),
            "habitaciones": rng.integers(1, 7, size=n),
            "antiguedad_anios": rng.integers(0, 51, size=n),
            "tiene_garaje": rng.integers(0, 2, size=n),
        }
    )
    price = (
        25000
        + 950 * df["metros_cuadrados"]
        + 12000 * df["habitaciones"]
        - 1300 * df["antiguedad_anios"]
        + 22000 * df["tiene_garaje"]
    )
    df[TARGET] = (price + rng.normal(0, 25000, size=n)).clip(lower=15000).round(0)
    return df


def build_pipeline() -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("scale", StandardScaler(), NUMERIC),
            ("binary", "passthrough", BINARY),
        ]
    )
    regressor = XGBRegressor(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_STATE,
    )
    return Pipeline([("preprocessor", preprocessor), ("regressor", regressor)])


def main() -> None:
    df = simulate_housing()
    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    metrics = {
        "model_name": "housing_model",
        "features": FEATURES,
        "numeric_scaled": NUMERIC,
        "n_samples": int(len(df)),
        "test": {
            "rmse": float(mean_squared_error(y_test, y_pred) ** 0.5),
            "mae": float(mean_absolute_error(y_test, y_pred)),
            "r2": float(r2_score(y_test, y_pred)),
        },
    }

    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(pipeline, MODELS_DIR / "housing_model.joblib")
    (MODELS_DIR / "housing_report.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )

    print(f"[housing] R2={metrics['test']['r2']:.4f} RMSE={metrics['test']['rmse']:.2f} MAE={metrics['test']['mae']:.2f}")
    print(f"[housing] Model saved at {MODELS_DIR / 'housing_model.joblib'}")


if __name__ == "__main__":
    main()

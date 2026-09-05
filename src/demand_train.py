import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"

TEST_SIZE = 0.2
RANDOM_STATE = 42

DAYS = ["Lunes", "Martes", "Miercoles", "Jueves", "Viernes", "Sabado", "Domingo"]
PRODUCTS = ["Electronica", "Ropa", "Alimentos", "Hogar", "Juguetes"]

FEATURES = ["dia_semana", "tipo_producto", "precio", "promocion_activa"]
CATEGORICAL = ["dia_semana", "tipo_producto"]
NUMERIC = ["precio", "promocion_activa"]
TARGET = "volumen_ventas"

DAY_WEIGHT = {
    "Lunes": 0.90, "Martes": 0.85, "Miercoles": 0.95, "Jueves": 1.00,
    "Viernes": 1.30, "Sabado": 1.60, "Domingo": 1.20,
}
PRODUCT_WEIGHT = {
    "Electronica": 1.30, "Ropa": 1.00, "Alimentos": 1.40,
    "Hogar": 0.90, "Juguetes": 1.10,
}


def simulate_sales(n: int = 20000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "dia_semana": rng.choice(DAYS, size=n),
            "tipo_producto": rng.choice(PRODUCTS, size=n),
            "precio": np.round(rng.uniform(5.0, 120.0, size=n), 2),
            "promocion_activa": rng.integers(0, 2, size=n),
        }
    )
    base = df["dia_semana"].map(DAY_WEIGHT) * df["tipo_producto"].map(PRODUCT_WEIGHT) * 40
    log_vol = np.log(base) - 0.012 * df["precio"] + 0.45 * df["promocion_activa"]
    log_vol += rng.normal(0.0, 0.15, size=n)
    df[TARGET] = np.maximum(np.exp(log_vol), 0).round()
    return df


def build_pipeline() -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
            ("num", "passthrough", NUMERIC),
        ]
    )
    regressor = XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_STATE,
    )
    return Pipeline([("preprocessor", preprocessor), ("regressor", regressor)])


def main() -> None:
    df = simulate_sales()
    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    metrics = {
        "model_name": "demand_model",
        "features": FEATURES,
        "categorical": CATEGORICAL,
        "numeric": NUMERIC,
        "n_samples": int(len(df)),
        "test": {
            "rmse": float(mean_squared_error(y_test, y_pred) ** 0.5),
            "mae": float(mean_absolute_error(y_test, y_pred)),
            "r2": float(r2_score(y_test, y_pred)),
        },
    }

    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(pipeline, MODELS_DIR / "demand_model.joblib")
    (MODELS_DIR / "demand_report.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )

    print(f"[demand] R2={metrics['test']['r2']:.4f} RMSE={metrics['test']['rmse']:.2f} MAE={metrics['test']['mae']:.2f}")
    print(f"[demand] Model saved at {MODELS_DIR / 'demand_model.joblib'}")


if __name__ == "__main__":
    main()

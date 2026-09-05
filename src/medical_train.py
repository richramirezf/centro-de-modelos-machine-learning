import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "KaggleV2-May-2016.csv"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

TEST_SIZE = 0.2
RANDOM_STATE = 42
CLASS_LABELS = ["Asiste (0)", "No-Show (1)"]

FEATURES = ["Age", "WaitTime_Days", "Scholarship", "Hipertension", "Diabetes", "SMS_received"]
TARGET = "No-show_num"


def load_noshow_data(csv_path: str | Path | None = None) -> pd.DataFrame:
    if csv_path is None:
        csv_path = DATA_PATH
    df = pd.read_csv(Path(csv_path))
    print(f"[medical] Loaded {len(df):,} rows from {Path(csv_path).name}")
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Datetime parsing, WaitTime_Days and clean invalid rows."""
    df = df.copy()

    df["ScheduledDay"] = pd.to_datetime(df["ScheduledDay"])
    df["AppointmentDay"] = pd.to_datetime(df["AppointmentDay"])
    df["WaitTime_Days"] = (
        df["AppointmentDay"].dt.normalize() - df["ScheduledDay"].dt.normalize()
    ).dt.days

    before = len(df)
    df = df[(df["Age"] >= 0) & (df["WaitTime_Days"] >= 0)]
    print(f"[medical] Dropped {before - len(df):,} rows with Age<0 or WaitTime_Days<0")

    df[TARGET] = df["No-show"].map({"Yes": 1, "No": 0})
    return df


def build_pipeline(scale_pos_weight: float) -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("scale", StandardScaler(), ["Age", "WaitTime_Days"]),
        ],
        remainder="passthrough",
    )
    classifier = CalibratedClassifierCV(
        estimator=XGBClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            eval_metric="logloss",
            random_state=RANDOM_STATE,
        ),
        method="sigmoid",
        cv=3,
    )
    return Pipeline([("preprocessor", preprocessor), ("classifier", classifier)])


def plot_confusion(matrix: list, samples: int, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(4.6, 4))
    im = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks([0, 1], CLASS_LABELS)
    ax.set_yticks([0, 1], CLASS_LABELS)
    ax.set_xlabel("Predicción")
    ax.set_ylabel("Real")
    ax.set_title(f"Matriz de confusión (entrenamiento)\nXGBoost No-Show — {samples} muestras")
    for i in range(2):
        for j in range(2):
            ax.text(
                j, i, matrix[i][j], ha="center", va="center",
                color="white" if matrix[i][j] > 4000 else "#0b3d2e", fontsize=13,
            )
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def train(csv_path: str | Path | None = None) -> dict:
    df = load_noshow_data(csv_path)
    df = build_features(df)

    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    print(f"[medical] Train={len(X_train):,}  Test={len(X_test):,}  No-show rate train={y_train.mean():.4f}")

    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    pipeline = build_pipeline(scale_pos_weight=scale_pos_weight)
    pipeline.fit(X_train, y_train)

    y_test_proba = pipeline.predict_proba(X_test)[:, 1]

    fpr, tpr, thresholds = roc_curve(y_test, y_test_proba)
    youden = tpr - fpr
    best_idx = int(youden.argmax())
    threshold = float(thresholds[best_idx])

    y_test_pred = (y_test_proba >= threshold).astype(int)
    test = {
        "threshold": threshold,
        "accuracy": float(accuracy_score(y_test, y_test_pred)),
        "roc_auc": float(roc_auc_score(y_test, y_test_proba)),
        "precision_no_show": float(precision_score(y_test, y_test_pred)),
        "recall_no_show": float(recall_score(y_test, y_test_pred)),
        "no_show_rate_test": float(y_test.mean()),
    }

    y_train_pred = (pipeline.predict_proba(X_train)[:, 1] >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_train, y_train_pred, labels=[0, 1]).ravel()
    report = {
        "model_name": "noshow_model",
        "class_labels": CLASS_LABELS,
        "features": FEATURES,
        "matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
        "accuracy": float(accuracy_score(y_train, y_train_pred)),
        "precision_class_1": float(precision_score(y_train, y_train_pred)),
        "recall_class_1": float(recall_score(y_train, y_train_pred)),
        "samples": int(len(y_train)),
        "test": test,
    }

    for key, value in test.items():
        print(f"[medical] test {key}: {value:.4f}")
    print(f"[medical] train acc: {report['accuracy']:.4f}  recall(No-Show): {report['recall_class_1']:.4f}")

    MODELS_DIR.mkdir(exist_ok=True)
    REPORTS_DIR.mkdir(exist_ok=True)
    joblib.dump(pipeline, MODELS_DIR / "noshow_model.joblib")
    (MODELS_DIR / "confusion_report_noshow.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    plot_confusion(report["matrix"], report["samples"], REPORTS_DIR / "confusion_noshow_model.png")
    print(f"[medical] Model saved at {MODELS_DIR / 'noshow_model.joblib'}")
    return report


if __name__ == "__main__":
    train()

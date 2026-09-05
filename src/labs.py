"""Laboratorios interactivos de clasificación (umbral, matriz de confusión
dinámica y curva ROC) usando arrays de test precalculados."""

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from src.academy_ui import render_concept
from sklearn.metrics import roc_auc_score, roc_curve


PROJECT_ROOT = Path(__file__).resolve().parent.parent
EVAL_DIR = PROJECT_ROOT / "models" / "eval"


def load_eval(name: str) -> tuple[np.ndarray, np.ndarray]:
    """Cached loader of (y_test, proba_clase_1) from models/eval/<name>.npz."""
    path = EVAL_DIR / f"{name}.npz"
    if not path.exists():
        st.error(f"Artefacto de evaluación no encontrado: {path.name}. Ejecuta `python src/eval_export.py`.")
        return np.array([]), np.array([])
    with np.load(path) as data:
        return data["y"].astype(int), data["p"].astype(float)


def _metrics_at(y: np.ndarray, p: np.ndarray, threshold: float) -> dict:
    pred = (p >= threshold).astype(int)
    tn = int(((y == 0) & (pred == 0)).sum())
    fp = int(((y == 0) & (pred == 1)).sum())
    fn = int(((y == 1) & (pred == 0)).sum())
    tp = int(((y == 1) & (pred == 1)).sum())
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    accuracy = (tn + tp) / max(len(y), 1)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    return {
        "TN": tn, "FP": fp, "FN": fn, "TP": tp,
        "precision": precision, "recall": recall,
        "specificity": specificity, "accuracy": accuracy, "f1": f1, "fpr": fpr,
    }


def render_threshold_lab(
    prefix: str,
    eval_name: str,
    *,
    model_label: str,
    neg_label: str,
    pos_label: str,
    default_threshold: float = 0.5,
    context: str = "",
) -> None:
    """Umbral + matriz de confusión dinámica + curva ROC sobre el split de test."""
    y, p = load_eval(eval_name)
    if len(y) == 0:
        return

    st.markdown(f"**Modelo analizado:** {model_label}")
    if context:
        st.caption(context)

    n_pos = int(y.sum())
    n_neg = int(len(y) - n_pos)
    st.caption(
        f"Split de test: {len(y):,} casos · {n_neg:,} clase negativa ({neg_label}) · "
        f"{n_pos:,} clase positiva ({pos_label}, {n_pos / len(y):.1%})."
    )

    threshold = st.slider(
        "Umbral de decisión (probabilidad)",
        min_value=0.0, max_value=1.0, value=default_threshold, step=0.01,
        key=f"{prefix}_threshold",
        help="Por encima de este valor la muestra se clasifica como clase positiva (riesgo).",
    )

    m = _metrics_at(y, p, threshold)

    col_cells = st.columns(4)
    col_cells[0].metric("TN", f"{m['TN']:,}", help=f"Negativos bien clasificados ({neg_label})")
    col_cells[1].metric("FP", f"{m['FP']:,}", help=f"Falsas alarmas ({neg_label} marcados como {pos_label})")
    col_cells[2].metric("FN", f"{m['FN']:,}", help=f"{pos_label} no detectados (el error más caro)")
    col_cells[3].metric("TP", f"{m['TP']:,}", help=f"{pos_label} detectados correctamente")

    st.dataframe(
        pd.DataFrame(
            [[m["TN"], m["FP"]], [m["FN"], m["TP"]]],
            index=[f"Real: {neg_label}", f"Real: {pos_label}"],
            columns=[f"Pred: {neg_label}", f"Pred: {pos_label}"],
        )
    )

    col_m1, col_m2, col_m3, col_m4, col_m5 = st.columns(5)
    col_m1.metric("Precision", f"{m['precision']:.3f}")
    col_m2.metric("Recall (TPR)", f"{m['recall']:.3f}")
    col_m3.metric("Specificity", f"{m['specificity']:.3f}")
    col_m4.metric("F1", f"{m['f1']:.3f}")
    col_m5.metric("Accuracy", f"{m['accuracy']:.3f}")

    st.markdown(
        f"Con umbral **{threshold:.2f}**, el modelo detecta **{m['recall']:.0%}** del riesgo real "
        f"({m['recall'] * 100:.0f}% de recall) con **{m['fpr']:.0%}** de falsas alarmas (FPR). "
        "Bajar el umbral sube recall pero también FP; subirlo reduce falsas alarmas pero deja pasar riesgo (FN)."
    )

    render_concept("threshold")

    st.markdown("#### Curva ROC (test)")
    if len(np.unique(y)) > 1:
        auc = float(roc_auc_score(y, p))
        fpr, tpr, _ = roc_curve(y, p)
        roc_df = pd.DataFrame({"FPR": fpr, "Modelo": tpr, "Azar": fpr})
        st.line_chart(roc_df, x="FPR", y=["Modelo", "Azar"], color=["#00897b", "#c3352b"])
        st.metric("ROC-AUC", f"{auc:.4f}", help="Área bajo la curva: capacidad de ordenar riesgo, independiente del umbral.")
        op = _metrics_at(y, p, threshold)
        st.caption(f"Punto operativo actual: TPR (recall) = {op['recall']:.3f} · FPR = {op['fpr']:.3f} a umbral {threshold:.2f}.")
    render_concept("roc_auc")


def render_pipeline_overview(model, *, title: str = "Pipeline en vivo") -> None:
    """Muestra los pasos y transformaciones reales de un pipeline cargado."""
    with st.expander(f"{title} (pasos y transformaciones)"):
        for i, (name, obj) in enumerate(model.steps, start=1):
            st.markdown(f"**{i}. `{name}`** — clase `{type(obj).__name__}`")
            if hasattr(obj, "get_params"):
                params = obj.get_params(deep=False)
                interesting = {k: v for k, v in params.items() if k in {
                    "C", "solver", "max_iter", "n_estimators", "learning_rate",
                    "max_depth", "scale_pos_weight", "eval_metric", "ngram_range",
                    "sublinear_tf", "method", "cv", "random_state",
                }}
                if interesting:
                    st.caption("Hiperparámetros: " + ", ".join(f"{k}={v}" for k, v in interesting.items()))

        preprocessor = model.named_steps.get("preprocessor")
        if preprocessor is not None and hasattr(preprocessor, "transformers_"):
            st.markdown("**Transformadores del preprocesador:**")
            for name, transformer, columns in preprocessor.transformers_:
                cols = columns if isinstance(columns, (list, tuple)) else [str(columns)]
                st.markdown(f"- `{name}` ({type(transformer).__name__}) → columnas: {', '.join(cols)}")
            try:
                names = list(preprocessor.get_feature_names_out())
                st.caption(f"El preprocesador genera **{len(names)}** columnas numéricas finales.")
            except Exception:
                pass

        vectorizer = model.named_steps.get("tfidf")
        if vectorizer is not None:
            try:
                feats = list(vectorizer.get_feature_names_out())
                st.caption(f"El vocabulario TF-IDF tiene **{len(feats):,}** términos.")
            except Exception:
                pass

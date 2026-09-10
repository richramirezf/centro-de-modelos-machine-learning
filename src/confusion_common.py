"""Utilidades compartidas para generar matrices de confusión y sus reportes."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score


def confusion_metrics(y_true, y_pred, labels=(0, 1)) -> dict:
    """Métricas estándar de una matriz de confusión binaria (clase 1 = positiva)."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=list(labels)).ravel()
    return {
        "matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_class_1": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall_class_1": float(recall_score(y_true, y_pred, zero_division=0)),
        "samples": int(len(y_true)),
    }


def plot_confusion(
    report: dict,
    output_path: Path,
    *,
    labels: list[str],
    title: str,
) -> None:
    """Dibuja y guarda la matriz de confusión de un reporte."""
    matrix = report["matrix"]
    fig, ax = plt.subplots(figsize=(4.6, 4))
    im = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks([0, 1], labels)
    ax.set_yticks([0, 1], labels)
    ax.set_xlabel("Predicción")
    ax.set_ylabel("Real")
    ax.set_title(f"Matriz de confusión (entrenamiento)\n{title} — {report['samples']} muestras")

    for i in range(2):
        for j in range(2):
            ax.text(
                j, i, matrix[i][j],
                ha="center", va="center",
                color="white" if matrix[i][j] > 40 else "#0b3d2e",
                fontsize=14,
            )

    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

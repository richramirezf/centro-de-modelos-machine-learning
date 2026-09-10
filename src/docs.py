"""Renderizadores de documentación compartidos por las páginas de ejercicios."""

from pathlib import Path

import pandas as pd
import streamlit as st


def render_features_table(rows: list[tuple], note: str | None = None) -> None:
    """Tabla de variables predictoras con columnas estándar.

    ``rows`` es una lista de tuplas ``(Variable, Tipo, Valores/Rango, Descripción)``.
    """
    df = pd.DataFrame(rows, columns=["Variable", "Tipo", "Valores / Rango", "Descripción"])
    st.dataframe(df, hide_index=True)
    if note:
        st.caption(note)


def render_model_doc(
    *,
    general: str,
    technical: str,
    metrics: dict[str, str],
    use_cases: str,
    considerations: str,
    title: str | None = None,
) -> None:
    """Tarjeta estándar de documentación de un modelo.

    ``metrics`` es un diccionario ``{etiqueta: valor}`` que se muestra como
    columnas de ``st.metric``.
    """
    if title:
        st.markdown(f"### {title}")
    with st.container(border=True):
        st.markdown("**¿Qué hace en general?**")
        st.write(general)
        st.markdown("**¿Cómo funciona técnicamente?**")
        st.write(technical)
        columns = st.columns(len(metrics))
        for column, (label, value) in zip(columns, metrics.items()):
            column.metric(label, value)
        st.markdown("**Casos de uso adicionales**")
        st.write(use_cases)
        st.markdown("**Consideraciones**")
        st.write(considerations)


def render_confusion_matrix(
    report: dict,
    *,
    labels: list[str],
    image_path: str | Path | None = None,
    caption: str | None = None,
    positive_name: str = "clase 1",
) -> None:
    """Matriz de confusión (imagen opcional + tabla + métricas resumidas).

    ``report`` debe incluir ``matrix``, ``TN/FP/FN/TP``, ``accuracy``,
    ``precision_class_1``, ``recall_class_1`` y ``samples``.
    """
    if image_path is not None:
        st.image(str(image_path), width="stretch", caption=caption)
    st.dataframe(
        pd.DataFrame(
            report["matrix"],
            index=[f"Real: {labels[0]}", f"Real: {labels[1]}"],
            columns=[f"Pred: {labels[0]}", f"Pred: {labels[1]}"],
        )
    )
    st.write(
        f"Aciertos: {report['TN'] + report['TP']} de {report['samples']} "
        f"(accuracy {report['accuracy']:.4f}) · "
        f"Precision ({positive_name}) {report['precision_class_1']:.4f} · "
        f"Recall ({positive_name}) {report['recall_class_1']:.4f}"
    )


def render_confusion_theory(pos_event: str) -> None:
    """Generic confusion-matrix theory. ``pos_event`` names the positive class."""
    st.markdown(
        f"""
        La matriz de confusión resume **aciertos y errores** de un clasificador binario.
        En este ejercicio la clase **positiva es `{pos_event}`**, el evento de riesgo que queremos detectar:

        - **TN — Verdadero Negativo:** caso sin evento correctamente clasificado. *Resultado sano y sin fricción.*
        - **FP — Falso Positivo:** caso sano marcado como riesgo por error (falsa alarma). *Costo de oportunidad: tratamos como riesgo a quien no lo es.*
        - **FN — Falso Negativo:** caso de riesgo que no se detecta (el error más caro). *El evento ocurre sin que hayamos actuado.*
        - **TP — Verdadero Positivo:** caso de riesgo detectado correctamente. *Riesgo identificado a tiempo para intervenir.*

        **Métricas derivadas:**
        - **Accuracy** = (TN + TP) / Total — aciertos globales; engaña con clases desbalanceadas.
        - **Precision (clase positiva)** = TP / (TP + FP) — de lo que señalé como riesgo, cuánto era riesgo real.
        - **Recall / Sensibilidad (clase positiva)** = TP / (TP + FN) — del riesgo real, cuánto logré detectar.
        - **Specificity** = TN / (TN + FP) — de los casos sanos, cuántos se clasificaron bien.
        - **F1** = media armónica de precision y recall.
        - **FPR (False Positive Rate)** = FP / (FP + TN) — casos sanos marcados como riesgo.

        **¿Por qué importa el umbral?** El modelo nunca "decide" solo: emite una probabilidad y nosotros fijamos el punto de corte.
        Bajar el umbral marca más casos como riesgo (suben TP y FP): detectamos más eventos pero aumentamos las falsas alarmas.
        Subirlo marca menos (suben TN y FN): reducimos fricción pero dejamos pasar más riesgo.
        El deslizador de **"Apetito de Riesgo"** (o el criterio de negocio del ejercicio) materializa esa decisión de costos.
        """
    )


def render_standard(note: str) -> None:
    """Ideal industry-standard (CRISP-DM + MLOps) step-by-step."""
    steps = [
        ("1. Entendimiento del negocio", "Objetivo medible, definición del evento a predecir, unidad de análisis y aceptación regulatoria."),
        ("2. Recolección y entendimiento de datos", "Fuentes, población objetivo, ventana temporal, volumen y representatividad histórica."),
        ("3. Calidad y limpieza", "Nulos, valores atípicos, inconsistencias e imputación con criterio documentado."),
        ("4. Análisis exploratorio (EDA)", "Balance del target, correlaciones, perfil de riesgo por atributo e hipótesis de negocio."),
        ("5. Ingeniería de atributos", "Codificación de categóricas, escalado de numéricas y creación de variables de negocio cuando aplique."),
        ("6. Diseño experimental", "Split estratificado y ventanas out-of-time; validación cruzada (StratifiedKFold); baseline simple de referencia."),
        ("7. Entrenamiento", "Comparar familias de modelos (lineal vs ensamble) con pipelines idénticos e hiperparámetros controlados."),
        ("8. Evaluación", "Métricas discriminativas (ROC-AUC), de calibración y de negocio; matriz de confusión y curvas ROC/PR. Nunca decidir solo con accuracy."),
        ("9. Calibración del umbral", "Fijar el corte de probabilidad según apetito de riesgo y costos FN vs FP."),
        ("10. Validación y riesgo", "Out-of-time, estabilidad de variables, sesgo y deriva (drift)."),
        ("11. Despliegue y monitoreo", "Exposición vía API/dashboard, versionado de modelo y datos, monitoreo de rendimiento y reentrenamiento programado."),
        ("12. Gobernanza y documentación", "Interpretabilidad (coeficientes / SHAP), trazabilidad de decisiones y auditoría regulatoria."),
    ]
    st.markdown(
        "El estándar de referencia para proyectos de analítica/ML es **CRISP-DM**, y la práctica de industria lo extiende con MLOps. "
        "Secuencia recomendada:"
    )
    for title, desc in steps:
        st.markdown(f"- **{title}:** {desc}")

    st.info(note)

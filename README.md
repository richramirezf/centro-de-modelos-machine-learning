# Centro de Modelos de Machine Learning

Portal unificado de **modelos de machine learning** construido con Streamlit. Reúne **seis ejercicios** —clasificación binaria, regresión continua y NLP— con una **misma metodología**: cada ejercicio incluye su caso de negocio, predicción interactiva y una pestaña de *documentación* (variables predictoras, funcionamiento general/técnico, casos de uso, consideraciones, métricas y paso a paso estándar CRISP-DM + MLOps).

---

## Ejercicios incluidos

| Ejercicio | Dataset | Registros | Modelo(s) | Problema |
|---|---|---|---|---|
| **Scoring de Crédito** | German Credit | 1,000 | Regresión Logística + XGBoost | Clasificación: default crediticio |
| **Churn Telco** | Telco Customer Churn (IBM) | 7,043 | XGBoost (+ SHAP) | Clasificación: abandono de clientes |
| **Ausentismo Médico (No-Show)** | Medical Appointment No Shows | 110,527 | XGBoost calibrado | Clasificación: inasistencia a citas |
| **Pronóstico de Demanda** | Simulado (ventas) | 20,000 | XGBRegressor | Regresión: volumen de ventas |
| **Valuación Inmobiliaria** | Simulado (inmobiliario) | 15,000 | XGBRegressor | Regresión: precio de propiedad (USD) |
| **Clasificador de Textos (NLP)** | Simulado (chat en español) | 522 | TF-IDF + Regresión Logística | NLP: intención de mensaje |

---

## Arquitectura del proyecto

```
centro-de-modelos-machine-learning/
├── app.py                        # Router multipágina (st.navigation)
├── app_pages/
│   ├── inicio.py                 # Hub con tarjetas de los 6 ejercicios
│   ├── credit_scoring.py         # Ejercicio 1: scoring de crédito
│   ├── telco_churn.py            # Ejercicio 2: churn de telecomunicaciones
│   ├── noshow.py                 # Ejercicio 3: ausentismo médico (No-Show)
│   ├── demand.py                 # Ejercicio 4: pronóstico de demanda
│   ├── housing.py                # Ejercicio 5: valuación inmobiliaria
│   └── intent.py                 # Ejercicio 6: clasificador de textos (NLP)
├── src/                          # Módulos de entrenamiento y documentación
│   ├── data_loader.py            # Carga del German Credit Dataset
│   ├── quality.py                # Imputación y codificación del target
│   ├── eda.py                    # Análisis exploratorio
│   ├── train.py                  # Entrenamiento comparativo LogReg / XGBoost
│   ├── confusion.py              # Matrices de confusión de entrenamiento (crédito)
│   ├── medical_train.py          # Pipeline de datos + modelo No-Show (calibrado)
│   ├── demand_train.py           # Regresión de demanda (simulado) → demand_model
│   ├── housing_train.py          # Regresión inmobiliaria (simulado) → housing_model
│   ├── nlp_train.py              # Clasificación de intenciones NLP → nlp_intent_model
│   ├── api.py                    # Microservicio FastAPI (crédito)
│   └── docs.py                   # Tema + documentación compartida (teoría + CRISP-DM)
├── src_telco/                    # Módulos del ejercicio de churn
│   ├── data_loader.py
│   ├── quality.py
│   └── train.py
├── models/                       # Pipelines serializados (.joblib) + reportes JSON
│   ├── logistic_model.joblib
│   ├── xgb_model.joblib
│   ├── churn_model.joblib
│   ├── noshow_model.joblib
│   ├── demand_model.joblib
│   ├── housing_model.joblib
│   ├── nlp_intent_model.joblib
│   ├── confusion_report.json        # (crédito)
│   ├── confusion_report_telco.json  # (churn)
│   ├── confusion_report_noshow.json # (no-show)
│   ├── demand_report.json           # (regresión demanda)
│   ├── housing_report.json          # (regresión vivienda)
│   └── nlp_intent_report.json       # (NLP intenciones)
├── reports/                      # Imágenes de matrices de confusión (PNG)
├── data/                         # Datasets locales (GITIGNORED, ver sección Datos)
└── .streamlit/config.toml        # Tema Teal / Material 3
```

---

## Stack tecnológico

| Componente | Tecnología |
|---|---|
| Interfaz | Streamlit (multipágina `st.navigation`) |
| Datos | pandas, numpy |
| Machine Learning | scikit-learn, XGBoost |
| Explicabilidad | SHAP |
| Visualización | Plotly, matplotlib |
| Microservicio | FastAPI + Pydantic v2 + Uvicorn |
| Serialización | joblib |

---

## Ejercicio 1 — Scoring de Crédito

**Problema:** estimar la probabilidad de *default* (`Risk`: good/bad) de una solicitud de crédito.

- **Variables predictoras:** `Age`, `Sex`, `Housing`, `Saving accounts`, `Checking account`, `Credit amount`, `Duration`, `Purpose`. Target binario codificado como `Risk_num` (bad = 1).
- **Pipeline:** `ColumnTransformer` (OneHotEncoder sobre categóricas + StandardScaler sobre numéricas) + clasificador.
- **Modelos:** Regresión Logística (interpretable) y XGBoost.
- **Interacción:** umbral dinámico — deslizador *"Apetito de Riesgo"* (10–90%) que decide aprobar/denegar según `predict_proba`.
- **Microservicio FastAPI** (`src/api.py`): carga los `.joblib` en memoria vía `lifespan` y expone `POST /predict/?model_type=logistic|xgboost` con payload Pydantic estricto (`CreditApplication`) + `GET /health`.

### Métricas reportadas

| Modelo | ROC-AUC (test) | Recall clase 1 (test) | Accuracy (entrenamiento) |
|---|---|---|---|
| Regresión Logística | 0.7615 | 0.3833 | 0.7562 |
| XGBoost | 0.7440 | 0.5333 | 1.0000 (memoriza entrenamiento) |

---

## Ejercicio 2 — Churn de Telecomunicaciones

**Problema:** estimar la probabilidad de que un cliente abandone (`Churn`: Yes/No) para intervención de retención.

- **Variables predictoras (19):** 3 numéricas (`tenure`, `MonthlyCharges`, `TotalCharges`) + 16 categóricas de contrato/servicios. Se descarta `customerID`.
- **Calidad de datos:** coerción de `TotalCharges` a numérico e imputación por mediana (11 nulos).
- **Pipeline:** OneHotEncoder + StandardScaler + `XGBClassifier` (200 árboles, `max_depth=5`, `learning_rate=0.1`).
- **Interacción:** EDA interactivo (Plotly), predicción por cliente con **explicabilidad SHAP** (waterfall), predicción masiva por CSV con descarga de resultados.
- Módulos de entrenamiento bajo `src_telco/` para no colisionar con los del crédito.

### Métricas reportadas

| Conjunto | Accuracy | ROC-AUC | Precision (churn) | Recall (churn) |
|---|---|---|---|---|
| Entrenamiento | 0.8791 | — | 0.8200 | 0.6977 |
| Test | 0.7942 | 0.8338 | 0.6429 | 0.5053 |

---

## Ejercicio 3 — Ausentismo Médico (No-Show)

**Problema:** estimar la probabilidad de que un paciente no asista a su cita (`No-show`: Yes/No).

- **Calidad de datos:** `ScheduledDay` / `AppointmentDay` a datetime; se crea `WaitTime_Days` (diferencia en días); se eliminan filas con `Age < 0` o `WaitTime_Days < 0` (6 filas).
- **Variables predictoras (6):** `Age`, `WaitTime_Days`, `Scholarship`, `Hipertension`, `Diabetes`, `SMS_received`. Target binario (Yes = 1).
- **Pipeline:** StandardScaler (numéricas) + **`CalibratedClassifierCV` (sigmoid)** envolviendo `XGBClassifier` (200 árboles, `learning_rate=0.05`, `max_depth=4`, `scale_pos_weight≈4.9`). La calibración evita probabilidades infladas frente al desbalance (≈20% no-show); punto de decisión por criterio de **Youden** (≈19%).
- **Interacción:** ficha del paciente (edad, días de espera, condiciones y SMS) → alerta visual por bandas de riesgo (<25% verde · 25–40% ámbar · ≥40% rojo) sobre `predict_proba`.

### Métricas reportadas (test, umbral Youden ≈ 0.19)

| Accuracy | ROC-AUC | Precision (No-Show) | Recall (No-Show) |
|---|---|---|---|
| 0.5802 | 0.7227 | 0.3037 | 0.8351 |

> La accuracy baja refleja el balanceo hacia la clase minoritaria: para decidir se usan ROC-AUC y recall/precision de la clase No-Show (baseline "siempre asiste" ≈ 80%).

---

## Ejercicios 4-6 — Regresión y NLP (datos simulados)

### 4 — Pronóstico de Demanda (`src/demand_train.py` → `models/demand_model.joblib`)

- **Problema:** regresión continua — estimar `volumen_ventas` (unidades) a partir de `dia_semana`, `tipo_producto`, `precio` y `promocion_activa`.
- **Pipeline:** ColumnTransformer (OneHotEncoder sobre categóricas) + `XGBRegressor` (300 árboles, `lr=0.05`, `max_depth=5`).
- **Métricas (test):** R² = 0.9117 · RMSE = 5.87 · MAE = 4.08.
- **Vista:** selectores de día/producto, precio numérico y checkbox de promoción; salida `st.metric` con unidades proyectadas.

### 5 — Valuación Inmobiliaria (`src/housing_train.py` → `models/housing_model.joblib`)

- **Problema:** regresión continua — estimar `precio_usd` a partir de `metros_cuadrados`, `habitaciones`, `antiguedad_anios` y `tiene_garaje`.
- **Pipeline:** ColumnTransformer (StandardScaler sobre numéricas) + `XGBRegressor` (400 árboles, `lr=0.05`, `max_depth=5`).
- **Métricas (test):** R² = 0.9011 · RMSE ≈ $25,309 · MAE ≈ $20,298.
- **Vista:** sliders de m²/habitaciones/antigüedad y selector de garaje; salida de precio formateada en USD (+ precio por m²).

### 6 — Clasificador de Textos / Intenciones (`src/nlp_train.py` → `models/nlp_intent_model.joblib`)

- **Problema:** NLP multiclase — clasificar mensajes de chat en `Soporte_Tecnico`, `Ventas`, `Reclamos`, `Horarios`.
- **Pipeline:** `TfidfVectorizer` (n-gramas 1–2, sublinear_tf) + `LogisticRegression` multinomial.
- **Métricas (test):** Accuracy = 1.0000 (corpus simulado con separación léxica clara; en producción se requiere más variedad).
- **Vista:** `st.text_area` para el mensaje; muestra intención detectada + confianza (`predict_proba`) y distribución de probabilidades.

---

## Metodología (CRISP-DM + MLOps)

Los seis ejercicios siguen la misma secuencia estándar, documentada en la pestaña de cada ejercicio:

1. Entendimiento del negocio → 2. Recolección de datos → 3. Calidad y limpieza → 4. EDA → 5. Ingeniería de atributos → 6. Diseño experimental (split 80/20 estratificado) → 7. Entrenamiento → 8. Evaluación (ROC-AUC, recall/precision, matriz de confusión) → 9. Calibración del umbral → 10. Validación y riesgo → 11. Despliegue y monitoreo → 12. Gobernanza y documentación.

---

## Instalación

```bash
git clone https://github.com/richramirezf/centro-de-modelos-machine-learning.git
cd centro-de-modelos-machine-learning
python -m venv venv
venv\Scripts\activate        # Windows  |  source venv/bin/activate (Linux/Mac)
pip install -r requirements.txt   # opcional, si se mantiene archivo de dependencias
```

Dependencias principales: `streamlit`, `pandas`, `numpy`, `scikit-learn`, `xgboost`, `joblib`, `matplotlib`, `plotly`, `shap`, `fastapi`, `uvicorn`, `pydantic`.

---

## Uso

### Ejecutar el portal (3 ejercicios)

```bash
python -m streamlit run app.py
```

Abre http://localhost:8501. La navegación por sidebar permite alternar entre el hub y cada ejercicio.

### Levantar la API de crédito (opcional)

```bash
python -m uvicorn src.api:app --host 127.0.0.1 --port 8600
```

Docs interactivos en http://127.0.0.1:8600/docs

### Reentrenar / regenerar reportes

```bash
python -m src.train           # crédito: entrena LogReg + XGBoost (models/*.joblib)
python -m src.confusion       # crédito: matrices de confusión (reports/ + JSON)
python -m src_telco.train     # churn: entrena XGBoost
python -m src.confusion_telco # churn: matrices de confusión
python -m src.medical_train   # no-show: pipeline + matriz de confusión
python src/demand_train.py    # regresión de demanda (genera demand_model.joblib)
python src/housing_train.py   # regresión inmobiliaria (genera housing_model.joblib)
python src/nlp_train.py       # NLP intenciones (genera nlp_intent_model.joblib)
```

---

## Datos

Los datasets NO se versionan (`.gitignore` excluye `*.csv`). Se esperan en `data/`:

| Archivo | Origen |
|---|---|
| `german_credit_data.csv` | Mirror público del German Credit Dataset |
| `WA_Fn-UseC_-Telco-Customer-Churn.csv` | Telco Customer Churn (IBM, Kaggle) |
| `KaggleV2-May-2016.csv` | Medical Appointment No Shows (Kaggle) |

El de no-show puede descargarse con `kagglehub.dataset_download("joniarroba/noshowappointments")`.

Los datasets de **demanda, vivienda y NLP son simulados** de forma determinista dentro de cada script de entrenamiento (`simulate_sales`, `simulate_housing`, `build_corpus`), por lo que no requieren archivos externos.

---

## Despliegue del modelo de crédito (ejemplo de payload)

```bash
curl -X POST "http://127.0.0.1:8600/predict/?model_type=xgboost" \
  -H "Content-Type: application/json" \
  -d '{"Age":35,"Sex":"male","Housing":"own","Saving accounts":"little",
       "Checking account":"moderate","Credit amount":3000,"Duration":24,"Purpose":"car"}'
```

Respuesta: `{model_type, probability, probability_percent, suggested_class, verdict}`.

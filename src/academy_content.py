"""Contenido educativo del portal (datos puros): glosario, conceptos,
guía de elección de modelo, ruta de aprendizaje, quizzes y FAQ."""

GLOSSARY_CATEGORIES = [
    "Tipos de aprendizaje",
    "Modelos",
    "Preprocesamiento",
    "Evaluación",
    "NLP",
    "Conceptos clave",
]

GLOSSARY = [
    {"term": "Aprendizaje supervisado", "category": "Tipos de aprendizaje",
     "definition": "Entrenar un modelo con ejemplos etiquetados (entrada + respuesta conocida) para que aprenda a predecir la respuesta de datos nuevos."},
    {"term": "Clasificación", "category": "Tipos de aprendizaje",
     "definition": "Problema donde la salida es una categoría discreta. Binaria: 2 clases (ej. default sí/no). Multiclase: más de 2 (ej. 4 intenciones)."},
    {"term": "Regresión", "category": "Tipos de aprendizaje",
     "definition": "Problema donde la salida es un número continuo (ej. precio, volumen de ventas). Se evalúa con R2/RMSE/MAE."},
    {"term": "Target / etiqueta", "category": "Conceptos clave",
     "definition": "La variable que se quiere predecir (columna objetivo). En el sitio: Risk, Churn, No-show, precio_usd, volumen_ventas o intención."},
    {"term": "Feature / variable predictora", "category": "Conceptos clave",
     "definition": "Cada atributo de entrada que usa el modelo. Ej.: Age, Duration, tenure, metros_cuadrados."},
    {"term": "Clase positiva / minoritaria", "category": "Conceptos clave",
     "definition": "En clasificación binaria, la clase de riesgo o evento que más interesa detectar (bad, churn, no-show) aunque sea la menos frecuente."},
    {"term": "Dataset balanceado vs desbalanceado", "category": "Conceptos clave",
     "definition": "Un target está desbalanceado cuando una clase domina. Ej. no-show ≈20% o churn ≈27%: el baseline 'predecir la clase mayoritaria' ya da alta exactitud."},
    {"term": "Baseline", "category": "Evaluación",
     "definition": "Modelo trivial de referencia (ej. predecir siempre la clase mayoritaria o la media). Un modelo real debe superarlo."},
    {"term": "Pipeline", "category": "Preprocesamiento",
     "definition": "Cadena de pasos (transformaciones + modelo) que se ejecutan juntos. Garantiza que preprocesamiento y predicción sean consistentes."},
    {"term": "One-Hot Encoding", "category": "Preprocesamiento",
     "definition": "Convierte cada categoría en una columna binaria. Ej. Housing 'own/rent/free' pasa a 3 columnas 0/1."},
    {"term": "StandardScaler", "category": "Preprocesamiento",
     "definition": "Estandariza variables numéricas: x'=(x-media)/desviación. Útil para modelos sensibles a la escala y para la interpretación."},
    {"term": "Train/test split", "category": "Evaluación",
     "definition": "Dividir datos en entrenamiento (el modelo aprende) y prueba (evalúa con datos no vistos). En el sitio: 80/20."},
    {"term": "Estratificación (stratify)", "category": "Evaluación",
     "definition": "Repartir el target con la misma proporción de clases en train y test. Clave con clases desbalanceadas."},
    {"term": "Overfitting (sobreajuste)", "category": "Conceptos clave",
     "definition": "El modelo memoriza el entrenamiento y falla en datos nuevos (brecha grande train vs test). XGBoost en datasets pequeños lo sufre: se decide con métricas de test."},
    {"term": "Underfitting", "category": "Conceptos clave",
     "definition": "El modelo es demasiado simple y no captura el patrón: rinde mal tanto en train como en test."},
    {"term": "Regresión Logística", "category": "Modelos",
     "definition": "Modelo lineal de clasificación. Estima P(clase)=sigmoide(w·x+b). Sus coeficientes son interpretables (peso de cada variable)."},
    {"term": "Gradient Boosting / XGBoost", "category": "Modelos",
     "definition": "Ensamble de árboles entrenados en secuencia; cada árbol corrige los errores del anterior. Potente con datos tabulares, pero menos interpretable."},
    {"term": "XGBRegressor", "category": "Modelos",
     "definition": "Variante de XGBoost para regresión continua (demanda y valuación inmobiliaria)."},
    {"term": "TF-IDF", "category": "NLP",
     "definition": "Peso de cada palabra en un texto: frecuencia en el documento (TF) ajustada por cuán rara es en el corpus (IDF). Palabras raras y distintivas pesan más."},
    {"term": "TfidfVectorizer", "category": "NLP",
     "definition": "Transformador de sklearn que convierte textos en una matriz numérica TF-IDF lista para un modelo."},
    {"term": "n-grama", "category": "NLP",
     "definition": "Grupo de n palabras consecutivas. n-gramas de 1–2 permiten capturar frases ('no funciona', 'a que hora')."},
    {"term": "Confusion matrix", "category": "Evaluación",
     "definition": "Tabla 2x2 de aciertos y errores: TN, FP, FN, TP. Es la base de accuracy, precision, recall y specificity."},
    {"term": "Accuracy", "category": "Evaluación",
     "definition": "(TN+TP)/Total. Útil con clases balanceadas; engaña si una clase domina."},
    {"term": "Precision", "category": "Evaluación",
     "definition": "TP/(TP+FP). De lo que el modelo marcó como positivo, cuánto era positivo real. Baja = muchas falsas alarmas."},
    {"term": "Recall / Sensibilidad", "category": "Evaluación",
     "definition": "TP/(TP+FN). Del riesgo real, cuánto logró detectar. Baja = se escapan eventos."},
    {"term": "F1", "category": "Evaluación",
     "definition": "Media armónica de precision y recall. Resume ambos cuando hay desbalance."},
    {"term": "Curva ROC / AUC", "category": "Evaluación",
     "definition": "ROC grafica TPR vs FPR para todos los umbrales. AUC resume la capacidad de ordenar: 0.5 = azar, 1.0 = perfecto."},
    {"term": "Umbral de decisión", "category": "Evaluación",
     "definition": "Punto de corte sobre la probabilidad para decidir la clase. Bajo umbral = se marca más riesgo; alto = más conservador."},
    {"term": "predict vs predict_proba", "category": "Evaluación",
     "definition": "predict() devuelve la clase; predict_proba() devuelve la probabilidad continua (base del umbral y de la calibración)."},
    {"term": "Calibración", "category": "Evaluación",
     "definition": "Que la probabilidad predicha coincida con la frecuencia real (si predice 30%, ~30% ocurre). Se logra con CalibratedClassifierCV."},
    {"term": "scale_pos_weight", "category": "Modelos",
     "definition": "Hiperparámetro de XGBoost que pondera más la clase minoritaria para compensar el desbalance."},
    {"term": "SHAP", "category": "Conceptos clave",
     "definition": "Valores de Shapley: cuánto aporta cada variable a la predicción de una muestra concreta. Gráfico waterfall = explicación por caso."},
    {"term": "Coeficiente (LogReg)", "category": "Modelos",
     "definition": "Peso w de cada variable en el modelo lineal. Positivo = aumenta la probabilidad de la clase 1; negativo = la reduce."},
    {"term": "Validación cruzada (CV)", "category": "Evaluación",
     "definition": "Entrenar/evaluar en varios pliegues del mismo dataset para estimar el rendimiento de forma más estable."},
    {"term": "R²", "category": "Evaluación",
     "definition": "Proporción de varianza explicada por una regresión. 1 = ajuste perfecto; valores cercanos a 0 = apenas mejor que la media."},
    {"term": "RMSE / MAE", "category": "Evaluación",
     "definition": "Errores de regresión en las unidades del target. MAE es el error absoluto medio; RMSE penaliza más los errores grandes."},
    {"term": "Residuo", "category": "Evaluación",
     "definition": "Diferencia entre el valor real y el predicho (real - predicho). Analizar residuos ayuda a detectar sesgos del modelo."},
]

CONCEPTS = {
    "roc_auc": (
        "Curva ROC y AUC",
        """
        La curva ROC ordena las predicciones de probabilidad y, para **cada umbral**, grafica:
        TPR (verdaderos positivos detectados) vs FPR (falsas alarmas). El **AUC** es el área bajo esa curva
        y mide cuán bien el modelo **ordena** los casos (0.5 = azar, 1.0 = perfecto).

        En los ejercicios usamos ROC-AUC como métrica principal porque no depende del umbral que elijas después.
        La curva te permite ver el **equilibrio** entre detectar riesgo (TPR) y generar falsas alarmas (FPR).
        """
    ),
    "threshold": (
        "Umbral de decisión",
        """
        El modelo emite una **probabilidad**; el umbral decide la clase. Con probabilidad de default = 45% y umbral 50% se aprueba;
        con umbral 40% se deniega. Bajar el umbral detecta más riesgo (sube recall) pero aumenta falsas alarmas (baja precision).

        El deslizador de "Apetito de Riesgo" (crédito) y el laboratorio de umbral son exactamente esto: el umbral es una **decisión de negocio**, no del algoritmo.
        """
    ),
    "calibration": (
        "Calibración de probabilidades",
        """
        Un modelo está calibrado si "cuando predice 30% de probabilidad, el evento ocurre ~30% de las veces".
        XGBoost crudo suele estar **sobreconfiado** (ej. media 44% cuando la tasa real es 20%). Con CalibratedClassifierCV (sigmoid/isotonic)
        se re-mapean las probabilidades para que coincidan con la frecuencia real.

        En el ejercicio de No-Show usamos calibración: las bandas de riesgo (bajo <25%, alto ≥40%) solo tienen sentido con probabilidades calibradas.
        """
    ),
    "imbalance": (
        "Clases desbalanceadas",
        """
        Cuando una clase domina (ej. 80% 'asiste', 20% 'no-show'), un modelo que siempre predice la clase mayoritaria logra 80% de exactitud
        sin aprender nada. Por eso: se usa split **estratificado**, se pondera la clase minoritaria (scale_pos_weight) y se evalúa con
        **recall/precision/AUC** en lugar de solo accuracy.
        """
    ),
    "onehot": (
        "One-Hot Encoding",
        """
        Convierte valores categóricos en columnas binarias. 'Housing = rent' → columna 'Housing_rent' = 1 y el resto 0.
        Evita inventar un orden entre categorías (no hay '3 > 2 > 1' entre 'own', 'rent', 'free').

        Al hacer One-Hot, el número de columnas crece: por eso el pipeline lo aplica siempre igual en entrenamiento e inferencia.
        """
    ),
    "overfitting": (
        "Sobreajuste (overfitting)",
        """
        El modelo memoriza el entrenamiento en vez de generalizar. Señal típica: exactitud casi perfecta en train y claramente menor en test.
        XGBoost sobre datasets pequeños (1,000 filas) puede llegar a 100% en entrenamiento.

        Mitigación: validación cruzada, early stopping, limitar profundidad/learning_rate y decidir SIEMPRE con métricas de **test**.
        """
    ),
    "shap": (
        "Explicabilidad SHAP",
        """
        Los modelos de árboles (XGBoost) no muestran fácilmente 'por qué' decidieron. SHAP usa valores de Shapley (teoría de juegos)
        para repartir la predicción entre las variables: cuánto empuja cada una hacia arriba o hacia abajo desde la predicción base.

        El gráfico waterfall muestra, para **una muestra concreta**, las variables que más influyeron (positivas en rojo/negativas en azul).
        """
    ),
    "pipeline": (
        "Pipelines y consistencia",
        """
        Un Pipeline encadena preprocesamiento + modelo en un solo objeto. Ventajas:
        - La misma transformación se aplica al entrenar y al predecir (nunca se te olvida el escalado).
        - Se guarda todo junto en un .joblib y se despliega como una unidad.

        En el sitio, cada .joblib contiene el pipeline completo: preprocesador + clasificador/regresor.
        """
    ),
    "tfidf": (
        "TF-IDF en palabras simples",
        """
        Para clasificar textos, hay que convertirlos en números. TF-IDF da peso a cada palabra:
        - TF: cuántas veces aparece en el mensaje.
        - IDF: cuánto pesa por ser rara en todo el corpus.

        Así, 'no funciona' pesa mucho para detectar soporte técnico, mientras que palabras comunes ('hola', 'gracias') pesan poco.
        """
    ),
}

ROUTE = [
    {"order": 1, "id": "demanda", "title": "Pronóstico de Demanda", "page": "app_pages/demand.py",
     "level": "Básico", "type": "Regresión", "goal": "Primer contacto con regresión y pipelines.",
     "learn": ["Regresión continua y métricas R²/RMSE/MAE", "ColumnTransformer + OneHotEncoder", "predict() en regresión"]},
    {"order": 2, "id": "vivienda", "title": "Valuación Inmobiliaria", "page": "app_pages/housing.py",
     "level": "Básico", "type": "Regresión", "goal": "Escalado de variables y predicción de un precio.",
     "learn": ["StandardScaler sobre numéricas", "Efecto de cada atributo en el precio", "Lectura de errores en USD"]},
    {"order": 3, "id": "nlp", "title": "Clasificador de Textos (NLP)", "page": "app_pages/intent.py",
     "level": "Intermedio", "type": "NLP", "goal": "De texto crudo a categoría con confianza.",
     "learn": ["TF-IDF y n-gramas", "Regresión Logística multinomial", "predict_proba y confianza"]},
    {"order": 4, "id": "credito", "title": "Scoring de Crédito", "page": "app_pages/credit_scoring.py",
     "level": "Intermedio", "type": "Clasificación", "goal": "Clasificación binaria, comparación de modelos y umbral.",
     "learn": ["Matriz de confusión y sus métricas", "LogReg interpretable vs XGBoost", "Apetito de riesgo = umbral"]},
    {"order": 5, "id": "churn", "title": "Churn Telco", "page": "app_pages/telco_churn.py",
     "level": "Avanzado", "type": "Clasificación", "goal": "Clasificación con explicabilidad por muestra.",
     "learn": ["EDA con Plotly", "SHAP waterfall por cliente", "Predicción masiva por CSV"]},
    {"order": 6, "id": "noshow", "title": "Ausentismo Médico (No-Show)", "page": "app_pages/noshow.py",
     "level": "Avanzado", "type": "Clasificación", "goal": "Desbalance, calibración y decisión por bandas de riesgo.",
     "learn": ["Desbalance de clases y stratify", "CalibratedClassifierCV", "Umbral de Youden y bandas de riesgo"]},
]

CHOOSE_MODEL_INTRO = """
Para elegir modelo, primero define el **tipo de problema**:

1. **¿Qué quieres predecir?** Un número continuo (regresión), una categoría (clasificación) o una etiqueta a partir de *texto* (NLP).
2. **¿Qué pesa más: interpretabilidad o precisión?** Riesgo regulado/auditable → modelos lineales; competencia de precisión → ensambles.
3. **¿Cuál es la métrica de negocio?** No la confundas con la de optimización: recall de no-show, AUC de crédito, R² de precio...
4. **¿El dataset está desbalanceado?** Si una clase domina, no uses solo accuracy: pondera clases y usa precision/recall/AUC.
"""

CHOOSE_MODEL_TABLE = [
    {"Tipo de problema": "Clasificación binaria", "Modelo base sugerido": "Regresión Logística",
     "Cuándo": "Pocas variables, necesidad de explicar cada caso (auditoría, scorecards)",
     "A favor": "Coeficientes interpretables, rápido, buen baseline", "En contra": "Asume relaciones lineales",
     "Ejemplo en el sitio": "Scoring de Crédito"},
    {"Tipo de problema": "Clasificación binaria", "Modelo base sugerido": "XGBoost",
     "Cuándo": "Datos tabulares con interacciones no lineales; precisión por encima de todo",
     "A favor": "Alta precisión, maneja no linealidades", "En contra": "Menos interpretable (usa SHAP), riesgo de sobreajuste",
     "Ejemplo en el sitio": "Churn Telco y No-Show"},
    {"Tipo de problema": "Regresión continua", "Modelo base sugerido": "XGBRegressor",
     "Cuándo": "Predecir un valor numérico (precio, volumen, tiempo)",
     "A favor": "Captura patrones no lineales, robusto", "En contra": "Requiere escalado; menos interpretable que una regresión lineal",
     "Ejemplo en el sitio": "Demanda y Valuación Inmobiliaria"},
    {"Tipo de problema": "NLP (texto)", "Modelo base sugerido": "TF-IDF + Regresión Logística",
     "Cuándo": "Clasificar mensajes/correos cortos sin grandes recursos",
     "A favor": "Simple, interpretable, efectivo en textos cortos", "En contra": "Ignora el orden semántico (usar embeddings/LLM para casos complejos)",
     "Ejemplo en el sitio": "Clasificador de Intenciones"},
    {"Tipo de problema": "Clasificación desbalanceada", "Modelo base sugerido": "XGBoost calibrado",
     "Cuándo": "La clase de interés es minoritaria (churn, no-show, fraude)",
     "A favor": "scale_pos_weight + calibración dan probabilidades útiles", "En contra": "La exactitud global deja de ser la métrica clave",
     "Ejemplo en el sitio": "Ausentismo Médico (No-Show)"},
]

QUIZZES = {
    "credito": [
        {"q": "¿Qué mide mejor la capacidad de un modelo para ordenar riesgo, sin depender del umbral?",
         "opts": ["Accuracy", "ROC-AUC", "Precision", "F1"], "answer": 1,
         "why": "El AUC resume TPR vs FPR en todos los umbrales; accuracy depende del umbral y del balance de clases.",
         "concept": "roc_auc"},
        {"q": "Con probabilidad de default 45% y umbral 40%, ¿qué decide el modelo?",
         "opts": ["Aprobar", "Denegar", "Reentrenar", "Depende de la clase mayoritaria"], "answer": 1,
         "why": "45% > 40%: la probabilidad supera el umbral → denegar. El umbral es una decisión de negocio.",
         "concept": "threshold"},
        {"q": "¿Qué significa un Falso Negativo en crédito?",
         "opts": ["Denegar a un buen pagador", "Aprobar a un cliente que caerá en default", "Detectar a un moroso", "Aprobar a un buen pagador"], "answer": 1,
         "why": "FN = se predijo clase 0 (aprobado) pero el real era 1 (default). Es el error más caro: pérdida directa.",
         "concept": None},
        {"q": "¿Por qué XGBoost logra 100% de exactitud en entrenamiento del crédito y no en test?",
         "opts": ["Porque el test tiene más datos", "Por sobreajuste: memoriza el entrenamiento", "Porque usa otro target", "Por falta de estratificación"], "answer": 1,
         "why": "Los modelos de boosting sobre datasets pequeños memorizan. Las decisiones deben usar métricas de test.",
         "concept": "overfitting"},
        {"q": "En Regresión Logística, un coeficiente positivo en 'Saving accounts' significa que…",
         "opts": ["Esa categoría reduce el default", "Esa categoría aumenta la probabilidad de default", "La variable no importa", "El modelo está mal calibrado"], "answer": 1,
         "why": "En la clase 1 (bad/default), un coeficiente positivo empuja la probabilidad hacia arriba.",
         "concept": None},
    ],
    "churn": [
        {"q": "¿Cuál es la clase positiva en el ejercicio de churn?",
         "opts": ["Que el cliente se quede", "Que el cliente abandone (Churn=1)", "El género del cliente", "El contrato anual"], "answer": 1,
         "why": "El evento de riesgo a detectar es el abandono; retenerlo cuesta menos que captar uno nuevo.",
         "concept": None},
        {"q": "¿Qué es el 'waterfall' de SHAP?",
         "opts": ["Un gráfico de series de tiempo", "El aporte de cada variable a la predicción de una muestra", "Una matriz de correlación", "La curva ROC"], "answer": 1,
         "why": "SHAP reparte la predicción entre variables usando valores de Shapley.",
         "concept": "shap"},
        {"q": "El dataset tiene ~27% de churn. ¿Por qué no usar solo accuracy?",
         "opts": ["Porque es muy lento de calcular", "Porque predecir siempre 'No Churn' ya daría ~73% sin aprender nada", "Porque accuracy no existe en clasificación", "Porque XGBoost no la soporta"], "answer": 1,
         "why": "Con clases desbalanceadas, accuracy es engañosa: el baseline mayoritario ya rinde alto.",
         "concept": "imbalance"},
        {"q": "¿Qué variables se descartan en el preprocesamiento y por qué?",
         "opts": ["MonthlyCharges, por redundante", "customerID: identificador sin valor predictivo (riesgo de leakage)", "tenure, por ser numérica", "Contract, por categórica"], "answer": 1,
         "why": "Los IDs nominales no predicen y pueden causar data leakage.",
         "concept": None},
        {"q": "Al subir el umbral de decisión en churn, esperarías…",
         "opts": ["Más recall y más falsas alarmas", "Menos detección de churn (más FN) pero menos falsas alarmas", "Que el AUC cambie", "Que la curva ROC se mueva"], "answer": 1,
         "why": "Un umbral alto aprueba/marca menos casos: baja el recall, bajan los FP. El AUC no depende del umbral.",
         "concept": "threshold"},
    ],
    "noshow": [
        {"q": "¿Qué transformación se crea para capturar la demora entre agendar y la cita?",
         "opts": ["One-Hot sobre la fecha", "WaitTime_Days = AppointmentDay - ScheduledDay (en días)", "StandardScaler de la hora", "TF-IDF del motivo"], "answer": 1,
         "why": "La espera es un predictor clave de inasistencia y se deriva restando fechas normalizadas.",
         "concept": None},
        {"q": "La media de probabilidades predichas (≈20%) coincide con la tasa real. Eso indica que el modelo está…",
         "opts": ["Sobreajustado", "Calibrado", "Subajustado", "Desbalanceado"], "answer": 1,
         "why": "Coincidir la media predicha con la frecuencia real es el sello de una buena calibración.",
         "concept": "calibration"},
        {"q": "¿Por qué el modelo usa scale_pos_weight≈4.9?",
         "opts": ["Porque hay más no-shows que asistentes", "Para compensar que solo ~20% son no-show", "Para acelerar el entrenamiento", "Para eliminar nulos"], "answer": 1,
         "why": "Ponderar la clase minoritaria evita que el modelo ignore los no-shows.",
         "concept": "imbalance"},
        {"q": "Con calibración, ¿qué pasa con el umbral 0.5 para predecir la clase?",
         "opts": ["Sigue siendo el mejor", "Casi nadie supera 50%: conviene un umbral menor (ej. Youden ~19%)", "Se vuelve binario el problema", "No cambia nada en la decisión"], "answer": 1,
         "why": "Probabilidades calibradas alrededor de 20% rara vez superan 0.5; se elige un punto operativo con Youden.",
         "concept": "threshold"},
        {"q": "¿Por qué la accuracy (~58%) es menor que el baseline 'siempre asiste' (~80%)?",
         "opts": ["Porque el modelo es malo", "Por balanceo: detecta no-shows a costa de marcar asistentes como riesgo", "Por error de datos", "Porque falta estandarizar"], "answer": 1,
         "why": "Al priorizar la clase minoritaria aumentan los FP; la métrica de negocio es recall/precision de no-show.",
         "concept": "imbalance"},
    ],
    "demanda": [
        {"q": "¿Qué tipo de problema resuelve el modelo de demanda?",
         "opts": ["Clasificación binaria", "Regresión continua", "NLP multiclase", "Clustering"], "answer": 1,
         "why": "Predice un número continuo (volumen de ventas), no una categoría.",
         "concept": None},
        {"q": "¿Qué métrica indica la proporción de varianza explicada?",
         "opts": ["RMSE", "MAE", "R²", "AUC"], "answer": 2,
         "why": "R²=0.91 significa que el modelo explica ~91% de la variabilidad de las ventas.",
         "concept": None},
        {"q": "Si activas la promoción, ¿qué espera el modelo?",
         "opts": ["Menos unidades", "Más unidades (la promoción suma en la simulación)", "Las mismas", "Depende solo del día"], "answer": 1,
         "why": "La simulación incorpora un efecto positivo de promocion_activa y el modelo lo aprende.",
         "concept": None},
        {"q": "¿Por qué se usa OneHotEncoder para día y tipo de producto?",
         "opts": ["Para escalar los números", "Porque son categóricas sin orden natural", "Para reducir columnas", "Para convertir texto en pesos"], "answer": 1,
         "why": "No existe 'Viernes > Lunes'; One-Hot evita inventar un orden.",
         "concept": "onehot"},
    ],
    "vivienda": [
        {"q": "¿Cuál es el objetivo (target) del modelo inmobiliario?",
         "opts": ["Metros cuadrados", "Precio en USD", "Número de habitaciones", "Tipo de barrio"], "answer": 1,
         "why": "precio_usd es la variable continua a predecir.",
         "concept": None},
        {"q": "¿Qué hace StandardScaler en este pipeline?",
         "opts": ["Convierte categorías en binarias", "Centra y escala las variables numéricas a media 0 y desviación 1", "Elimina outliers", "Traduce el texto"], "answer": 1,
         "why": "El escalado iguala magnitudes (m² vs años) para que el modelo las trate de forma comparable.",
         "concept": None},
        {"q": "Según la simulación, ¿qué efecto tiene la antigüedad sobre el precio?",
         "opts": ["Lo aumenta", "Lo reduce", "No tiene efecto", "Solo afecta con garaje"], "answer": 1,
         "why": "La fórmula simula -1300 USD por año de antigüedad; el modelo aprende ese patrón.",
         "concept": None},
        {"q": "Un MAE de ~20k USD en un rango de precios amplio indica…",
         "opts": ["El modelo es perfecto", "El error promedio por propiedad es ~20k USD", "La precisión es del 20%", "Hay overfitting"], "answer": 1,
         "why": "MAE es el error absoluto promedio en las unidades del target.",
         "concept": None},
    ],
    "intent": [
        {"q": "¿Qué paso convierte el mensaje 'a que hora abren' en números?",
         "opts": ["StandardScaler", "TfidfVectorizer", "OneHotEncoder", "ColumnTransformer con escalado"], "answer": 1,
         "why": "TF-IDF transforma texto en una matriz de pesos por término.",
         "concept": "tfidf"},
        {"q": "¿Qué modelo decide la intención final?",
         "opts": ["XGBRegressor", "Regresión Logística multinomial", "XGBoost binario", "KMeans"], "answer": 1,
         "why": "LR multiclase estima P(intención | texto) para las 4 clases.",
         "concept": None},
        {"q": "La 'confianza' que mostramos es…",
         "opts": ["El accuracy del modelo", "La probabilidad máxima entre las clases (argmax de predict_proba)", "La longitud del texto", "El número de n-gramas"], "answer": 1,
         "why": "Se toma el mayor valor de predict_proba y se muestra en %.",
         "concept": None},
        {"q": "¿Por qué palabras como 'hola' pesan poco en TF-IDF?",
         "opts": ["Porque son largas", "Porque aparecen en casi todos los mensajes (poca IDF)", "Porque el modelo las ignora por regla", "Porque están en mayúsculas"], "answer": 1,
         "why": "Las palabras comunes en todo el corpus tienen bajo peso discriminativo.",
         "concept": "tfidf"},
        {"q": "Si un cliente escribe 'quiero que me devuelvan mi dinero', ¿qué intención esperas?",
         "opts": ["Horarios", "Reclamos", "Ventas", "Soporte_Tecnico"], "answer": 1,
         "why": "Palabras de devolución/dinero corresponden al patrón de reclamos.",
         "concept": None},
    ],
    "general": [
        {"q": "¿Qué diferencia hay entre clasificación y regresión?",
         "opts": ["Clasificación predice categorías; regresión predice números continuos", "Son lo mismo", "Regresión solo se usa con texto", "Clasificación no usa etiquetas"], "answer": 0,
         "why": "La salida define el tipo: discreta (clase) vs continua (valor).",
         "concept": None},
        {"q": "¿Qué es un pipeline en scikit-learn?",
         "opts": ["Una base de datos", "Cadena de transformaciones + modelo que se aplican juntas", "Un gráfico de red", "Un tipo de dataset"], "answer": 1,
         "why": "Encadena preprocesamiento y modelo en un objeto único, consistente en entrenamiento e inferencia.",
         "concept": "pipeline"},
        {"q": "Con clases desbalanceadas, la métrica más engañosa es…",
         "opts": ["ROC-AUC", "Accuracy", "Recall", "Precision"], "answer": 1,
         "why": "Un modelo que siempre predice la clase mayoritaria logra alta accuracy sin aprender.",
         "concept": "imbalance"},
        {"q": "¿Para qué sirve el split estratificado?",
         "opts": ["Para acelerar el entrenamiento", "Para preservar la proporción de clases en train y test", "Para eliminar nulos", "Para escalar variables"], "answer": 1,
         "why": "Con stratify, cada partición mantiene la misma distribución del target.",
         "concept": None},
        {"q": "¿Qué indica un recall alto de la clase de riesgo?",
         "opts": ["Pocas falsas alarmas", "Se detecta la mayor parte del riesgo real (pocos FN)", "El modelo es perfecto", "No hay desbalance"], "answer": 1,
         "why": "Recall = TP/(TP+FN): mide cuánto del riesgo real se logra detectar.",
         "concept": None},
        {"q": "¿Por qué conviene cargar los modelos con @st.cache_resource en Streamlit?",
         "opts": ["Para que el modelo se cargue una sola vez y no en cada rerun", "Para entrenar más rápido", "Para subir el modelo a la nube", "Para eliminar datos"], "answer": 0,
         "why": "El caché de recursos evita recargar .joblib pesados en cada interacción.",
         "concept": None},
    ],
}

QUIZ_EXERCISE_TITLES = [
    ("credito", "Scoring de Crédito"),
    ("churn", "Churn Telco"),
    ("noshow", "Ausentismo Médico (No-Show)"),
    ("demanda", "Pronóstico de Demanda"),
    ("vivienda", "Valuación Inmobiliaria"),
    ("intent", "Clasificador de Textos (NLP)"),
    ("general", "Quiz general (todos los temas)"),
]

FAQ = [
    {"q": "¿Por qué XGBoost tiene 100% de exactitud en entrenamiento del crédito?",
     "a": "Es sobreajuste: los ensambles de árboles pueden memorizar datasets pequeños. Por eso siempre evaluamos con el split de test y comparamos train vs test."},
    {"q": "¿Por qué en No-Show la exactitud baja si 'el modelo anda bien'?",
     "a": "Al balancear hacia la clase minoritaria (no-show) el modelo marca como riesgo a muchos asistentes (FP). La exactitud deja de ser útil; se miran recall/precision/AUC de la clase no-show."},
    {"q": "¿Cuándo uso predict() y cuándo predict_proba()?",
     "a": "predict() devuelve la clase final; predict_proba() devuelve la probabilidad continua, necesaria para umbrales dinámicos, bandas de riesgo y calibración."},
    {"q": "¿Un AUC de 0.72 es bueno o malo?",
     "a": "Es moderado: mejor que 0.5 (azar) y lejos de 1.0. En riesgo/abandono reales valores 0.7-0.8 son comunes; lo importante es comparar contra un baseline y decidir el umbral según costos."},
    {"q": "¿El dataset se puede 'contaminar' con el objetivo?",
     "a": "Sí: usar el ID u otra columna que ya contenga el resultado (o que se conozca solo después) filtra información (data leakage). Por eso se elimina customerID y se define el target con cuidado."},
    {"q": "¿Qué significa que un modelo esté bien calibrado?",
     "a": "Que la probabilidad predicha coincide con la frecuencia observada. Si predice 30% y eso ocurre ~30% de las veces, las bandas de riesgo son confiables."},
    {"q": "¿Por qué no entrenar siempre el modelo más complejo?",
     "a": "Complejidad = riesgo de sobreajuste, menor interpretabilidad y mayor costo. Un modelo lineal bien hecho puede superar a uno complejo cuando hay pocos datos o se exige auditar decisiones."},
    {"q": "¿Qué es un umbral operativo y por qué no siempre es 0.5?",
     "a": "Es el punto de corte que decide la clase según costos (FN vs FP). En desbalance o tras calibración, 0.5 suele ser inadecuado; se elige con Youden, curvas ROC o costos de negocio."},
]

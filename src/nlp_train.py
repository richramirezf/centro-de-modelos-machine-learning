import json
import random
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"

TEST_SIZE = 0.2
RANDOM_STATE = 42

INTENTS = ["Soporte_Tecnico", "Ventas", "Reclamos", "Horarios"]


def build_corpus() -> list[tuple[str, str]]:
    ventas_verbs = [
        "quiero comprar", "quiero contratar", "me interesa adquirir", "necesito cotizar",
        "deseo conocer el precio de", "puedo comprar", "me gustaria contratar",
    ]
    ventas_objs = [
        "el plan premium", "el plan basico", "internet dedicado", "la linea movil",
        "el paquete familiar", "el servicio de television", "su producto",
    ]
    ventas_tails = ["", " hoy", " lo antes posible", " para mi casa", " para mi negocio", " por favor"]
    ventas_questions = ["cuanto cuesta", "cuanto vale", "que precio tiene", "cual es el costo de"]

    horario_frames = [
        "a que hora abren", "a que hora cierran", "cual es el horario de atencion",
        "hasta que hora atienden", "cual es el horario", "a que hora abre la sucursal",
    ]
    horario_locs = ["", " hoy", " manana", " los sabados", " los domingos", " los festivos", " en fin de semana"]

    reclamos_starts = [
        "me cobraron de mas", "quiero que me devuelvan mi dinero", "estoy inconforme con",
        "quiero presentar una queja por", "me llego danado", "hay un cargo que no reconozco",
        "me llegaron dos facturas", "el servicio se corto sin aviso",
        "quiero hablar con un supervisor por", "la atencion fue pesima",
        "me bloquearon la cuenta sin razon", "no me dieron lo que pedi",
    ]
    reclamos_cont = ["", " en mi ultima factura", " en el cobro de este mes", " en la atencion recibida", " por el pedido que hice"]

    soporte_starts = [
        "mi internet no funciona", "la aplicacion no abre", "no puedo iniciar sesion",
        "tengo un error en el sistema", "mi correo no envia", "la pagina no carga",
        "no recibo las notificaciones", "como restablezco mi contrasena", "el wifi se desconecta",
        "mi dispositivo no conecta", "la cuenta esta bloqueada",
        "no me deja descargar la actualizacion", "tengo problemas con el software",
        "la pantalla se queda en negro",
    ]
    soporte_cont = ["", " desde ayer", " despues de actualizar", " cuando abro la aplicacion", " en mi celular", " en mi computadora", " desde que cambie de plan"]

    corpus: list[tuple[str, str]] = []

    for verb in ventas_verbs:
        for obj in ventas_objs:
            for tail in ventas_tails:
                corpus.append((f"{verb} {obj}{tail}", "Ventas"))
    for q in ventas_questions:
        for obj in ventas_objs:
            corpus.append((f"{q} {obj}?", "Ventas"))

    for frame in horario_frames:
        for loc in horario_locs:
            corpus.append((f"{frame}{loc}?", "Horarios"))

    for start in reclamos_starts:
        for cont in reclamos_cont:
            corpus.append((f"{start}{cont}.", "Reclamos"))

    for start in soporte_starts:
        for cont in soporte_cont:
            corpus.append((f"{start}{cont}.", "Soporte_Tecnico"))

    rng = random.Random(RANDOM_STATE)
    rng.shuffle(corpus)
    return corpus


def build_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)),
            ("clf", LogisticRegression(C=1.0, max_iter=2000, random_state=RANDOM_STATE)),
        ]
    )


def main() -> None:
    corpus = build_corpus()
    df = pd.DataFrame(corpus, columns=["texto", "intencion"])

    X_train, X_test, y_train, y_test = train_test_split(
        df["texto"], df["intencion"], test_size=TEST_SIZE, stratify=df["intencion"], random_state=RANDOM_STATE
    )

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    report_text = classification_report(y_test, y_pred, output_dict=True)
    metrics = {
        "model_name": "nlp_intent_model",
        "intents": INTENTS,
        "n_samples": int(len(df)),
        "classes_": pipeline.named_steps["clf"].classes_.tolist(),
        "test": {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "report": {k: v for k, v in report_text.items() if not k.startswith("macro") and not k.startswith("weighted")},
        },
    }

    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(pipeline, MODELS_DIR / "nlp_intent_model.joblib")
    (MODELS_DIR / "nlp_intent_report.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"[nlp] Test accuracy: {metrics['test']['accuracy']:.4f}")
    print(f"[nlp] Samples: {len(df)}  Classes: {pipeline.named_steps['clf'].classes_.tolist()}")
    print(f"[nlp] Model saved at {MODELS_DIR / 'nlp_intent_model.joblib'}")


if __name__ == "__main__":
    main()

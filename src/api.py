from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"

MODEL_FILES = {
    "logistic": "logistic_model.joblib",
    "xgboost": "xgb_model.joblib",
}


class CreditApplication(BaseModel):
    model_config = ConfigDict(strict=True, populate_by_name=True)

    Age: int = Field(ge=18, le=100)
    Sex: str
    Housing: str
    Saving_accounts: str = Field(alias="Saving accounts")
    Checking_account: str = Field(alias="Checking account")
    Credit_amount: int = Field(alias="Credit amount", ge=0)
    Duration: int = Field(ge=1, le=72)
    Purpose: str


class PredictionResult(BaseModel):
    model_type: Literal["logistic", "xgboost"]
    probability: float
    probability_percent: float
    suggested_class: int
    verdict: str


def _load_models() -> dict[str, object]:
    return {
        key: joblib.load(MODELS_DIR / filename)
        for key, filename in MODEL_FILES.items()
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.models = _load_models()
    yield
    app.state.models.clear()


app = FastAPI(
    title="Credit Scoring API",
    description="Microservicio de scoring crediticio: probabilidad de default y clase sugerida.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict/", response_model=PredictionResult)
def predict(application: CreditApplication, model_type: Literal["logistic", "xgboost"] = "logistic") -> PredictionResult:
    try:
        pipeline = app.state.models[model_type]
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Modelo '{model_type}' no disponible.")

    row = pd.DataFrame([application.model_dump(by_alias=True)])
    prob = float(pipeline.predict_proba(row)[0, 1])

    return PredictionResult(
        model_type=model_type,
        probability=prob,
        probability_percent=round(prob * 100, 2),
        suggested_class=1 if prob > 0.5 else 0,
        verdict="CRÉDITO DENEGADO" if prob > 0.5 else "CRÉDITO APROBADO",
    )
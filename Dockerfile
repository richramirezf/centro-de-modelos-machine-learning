# syntax=docker/dockerfile:1

FROM python:3.14-slim

# --- Entorno ---
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# libgomp1 lo requiere XGBoost; curl para el healthcheck
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependencias primero para aprovechar la caché de capas
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Código, modelos, reportes y datos
COPY . .

EXPOSE 8501 8600

HEALTHCHECK --interval=30s --timeout=5s --start-period=45s --retries=3 \
    CMD curl -fsS http://localhost:8501/_stcore/health || exit 1

# Portal Streamlit (la API se levanta con otro comando, ver docker-compose.yml)
CMD ["streamlit", "run", "app.py"]

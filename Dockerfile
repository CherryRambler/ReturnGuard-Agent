FROM python:3.11-slim

WORKDIR /app

# System deps needed to build/run xgboost + shap wheels.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/ backend/
COPY src/ src/
COPY models/ models/
COPY data/synthetic/ data/synthetic/

ENV RETURNGUARD_DATASET=synthetic
# Plain filename by default (always writable inside the container's own
# filesystem). Override to a mounted-disk path (e.g. /data/returnguard.db)
# in the host platform's env config if you attach persistent storage -
# see render.yaml.
ENV RETURNGUARD_DB_PATH=returnguard.db

EXPOSE 8000

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]

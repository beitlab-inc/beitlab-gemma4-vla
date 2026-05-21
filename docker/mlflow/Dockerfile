FROM python:3.11-slim

ARG MLFLOW_VERSION=3.12.0

RUN pip install --no-cache-dir \
    "mlflow==${MLFLOW_VERSION}" \
    psycopg2-binary

EXPOSE 5000

FROM ghcr.io/mlflow/mlflow:v3.18.0
RUN pip install --no-cache-dir "psycopg2-binary>=2.9" \
    && useradd --create-home --uid 10001 mlflow \
    && install -d -o mlflow -g mlflow /mlartifacts
# Sem root: o volume de artefatos novo herda o dono do /mlartifacts da imagem.
USER mlflow

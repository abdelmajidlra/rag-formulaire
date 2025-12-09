# IRCC RAG Backend

API FastAPI qui expose le moteur RAG `rag_formulaire`.

## Installation locale

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements.txt
```

## Lancement

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

## Configuration

Variables principales (via `.env` ou variables d'environnement) :

- `API_HOST`, `API_PORT`
- `ALLOWED_ORIGINS`
- `RAG_INDEX_PATH`, `RAG_DATA_PATH`, `LLM_MODEL_PATH`
- `ENABLE_AUTH` (bool)
- `AZURE_AD_TENANT_ID`, `AZURE_AD_CLIENT_ID`, `AZURE_AD_API_AUDIENCE`

## Tests

```bash
PYTHONPATH=backend/src:src pytest backend/tests
```

## Docker

```bash
docker build -f backend/docker/Dockerfile.api -t ircc-rag-api .
docker run -p 8000:8000 ircc-rag-api
```

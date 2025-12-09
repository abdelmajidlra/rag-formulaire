# Infra & Ops

## Docker Compose

```bash
cd infra
docker-compose up --build
```

Services disponibles :

- `api` : FastAPI exposant le moteur RAG.
- `llama` : backend LLaMA (Text Generation Inference) servant le modèle `meta-llama/Llama-3.1-8B-Instruct`.
- `web` : front-end Vite connecté à l'API.

Pour charger le modèle LLaMA privé, exportez un token Hugging Face avant de lancer :

```bash
export HF_TOKEN=hf_...
```

## Kubernetes

Manifests dans `infra/k8s`. Adapter les images et PVC avant déploiement.

## Ingestion

- `python infra/scripts/init_indexes.py` reconstruit les index RAG.
- `infra/scripts/cron_reindex.sh` peut être ajouté à cron (`crontab -e`) pour une exécution régulière.

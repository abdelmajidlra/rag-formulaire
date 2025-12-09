# Infra & Ops

## Docker Compose

```bash
cd infra
docker-compose up --build
```

## Kubernetes

Manifests dans `infra/k8s`. Adapter les images et PVC avant déploiement.

## Ingestion

- `python infra/scripts/init_indexes.py` reconstruit les index RAG.
- `infra/scripts/cron_reindex.sh` peut être ajouté à cron (`crontab -e`) pour une exécution régulière.

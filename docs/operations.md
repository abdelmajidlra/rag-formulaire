# Operations

## Déploiement backend

- Docker : `docker build -f backend/docker/Dockerfile.api -t ircc-rag-api . && docker run -p 8000:8000 ircc-rag-api`.
- Compose : `docker-compose up --build` depuis `infra/`.
- Kubernetes : appliquer les manifests `infra/k8s/*.yaml` après avoir poussé l'image dans un registre.

## Portail web

- `npm install && npm run dev` depuis `frontend/web-portal`.
- Configurer `VITE_API_BASE_URL` vers l'API.

## Régénération des index

- Script unique : `python infra/scripts/init_indexes.py`.
- Planification : ajouter `infra/scripts/cron_reindex.sh` à cron (ex: `0 2 * * 1`).

## Gestion des incidents

- Vérifier les logs API (conteneur ou journald) et ceux de la tâche cron dans `infra/logs`.
- Redémarrer le déploiement (`kubectl rollout restart deploy/ircc-backend`) ou le conteneur docker.

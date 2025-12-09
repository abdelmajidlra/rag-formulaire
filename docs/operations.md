# Operations

## Préparer LLaMA (inférence)

- **Token HF** : requis pour les modèles LLaMA privés (`export HF_TOKEN=hf_...`).
- **Local GPU** : ≥14 GB VRAM pour charger `meta-llama/Llama-3.1-8B-Instruct` (quantification 4-bit possible via `RAG_FORM_GEN_4BIT=true`).
- **Endpoint externe** : définissez `RAG_FORM_GEN_ENDPOINT` pour déléguer la génération à TGI/Ollama et éviter le chargement local.

## Déploiement backend

- Docker : `docker build -f backend/docker/Dockerfile.api -t ircc-rag-api . && docker run -p 8000:8000 ircc-rag-api` (ajoutez `-e RAG_FORM_GEN_ENDPOINT=...` si vous ciblez un endpoint LLaMA).
- Compose : `docker-compose up --build` depuis `infra/` (services `llama` TGI, `api`, `web`).
- Kubernetes : appliquer les manifests `infra/k8s/*.yaml` après avoir poussé l'image dans un registre; montez un PVC pour les index et fournissez le token HF via Secret.

## Portail web

- `npm install && npm run dev` depuis `frontend/web-portal`.
- Configurer `VITE_API_BASE_URL` vers l'API (ex: `http://localhost:8000`).

## Régénération des index

- Script unique : `python infra/scripts/init_indexes.py`.
- Planification : ajouter `infra/scripts/cron_reindex.sh` à cron (ex: `0 2 * * 1`).
  Pensez à synchroniser le dossier `data/` (manifest + index) sur le même volume que l'API.

## Démos et notebooks

- Notebook local complet : `notebooks/ircc-rag-llama.ipynb` (exporter `RAG_FORM_GEN_ENDPOINT`/`HF_TOKEN` si nécessaire, puis exécuter ingestion + requêtes).
- Notebook Colab : `notebooks/colab-ircc-rag-poc.ipynb` (GPU T4), idéal pour valider la chaîne avec ressources limitées.

## Gestion des incidents

- Vérifier les logs API (conteneur ou journald) et ceux de la tâche cron dans `infra/logs`.
- Redémarrer le déploiement (`kubectl rollout restart deploy/ircc-backend`) ou le conteneur docker.

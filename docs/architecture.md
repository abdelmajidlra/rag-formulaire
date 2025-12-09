# Architecture

## Contexte

Assistant interne pour formulaires IRCC : un moteur RAG existant (`rag_formulaire`) enrichi par une API FastAPI, un portail web et un bot Teams.

## Couches

- **Front** : SPA React (intranet) + Bot Teams (Bot Framework).
- **API** : FastAPI (`ircc-rag-api`) gère auth, CORS, exposition du pipeline.
- **IA/RAG** : Module Python `rag_formulaire` (retrieval hybride, rerank, génération locale).
- **Données** : PDFs IRCC, index BM25 + vecteurs, stockés localement ou sur PVC.

## Flux détaillé

```mermaid
flowchart TD
    subgraph Ingestion[Phase 1 · Ingestion]
        DL[Web crawler \n downloader.py] --> PD[Parser Docling \n parser_docling.py]
        PD --> CH[Chunking contextuel \n chunking.py]
        CH --> BM[BM25]:::index
        CH --> VE[Vectoriel ChromaDB]:::index
    end

    subgraph API[Phase 2 · Backend FastAPI]
        U[Utilisateur Web/Teams] --> FE[Front-end React/Bot]
        FE -->|HTTP JSON| EP[/`POST /ask`/]
        EP --> RP[Pré-traitement requête \n query_processing.py]
        RP --> RT[Retrieval hybride \n retrieval.py]
        RT --> RR[Reranking cross-encoder \n reranker.py]
        RR --> EV[Filtrage CRAG \n evaluation.py]
    end

    subgraph Generation[Phase 3 · Génération]
        EV -->|preuves solides| LL[LLaMA 3.x (local ou endpoint) \n llm.py]
        EV -->|preuves faibles| FB[Message de fallback]
        LL --> AR[Auto-réflexion + validation \n evaluation.py]
    end

    BM & VE --> RT
    AR --> FE
    FB --> FE

    classDef index fill:#fff3cd,stroke:#333,stroke-width:1px
```

**Lecture du flux :**
- Ingestion télécharge et parse les formulaires, puis alimente les index BM25 et ChromaDB.
- Les frontends (web intranet, bot Teams) appellent le backend FastAPI qui orchestre pré-traitement, retrieval hybride, reranking et garde-fous CRAG.
- Si les preuves sont suffisantes, la génération passe par LLaMA 3.x (chargé localement ou via un endpoint TGI/Ollama), sinon un message de fallback est retourné.

## Déploiement

- Local : `docker-compose` (API + web).
- Prod : manifests Kubernetes (backend, frontend, ingress) + PVC pour les index.

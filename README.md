# RAG Formulaire IRCC

<div align="center">

**Système de Recherche Générative Augmentée (RAG) en Français pour les Formulaires IRCC**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

</div>

---

## 📋 Table des Matières

- [Vue d'ensemble](#-vue-densemble)
- [Architecture du Système](#-architecture-du-système)
- [Fonctionnalités Principales](#-fonctionnalités-principales)
- [Installation](#-installation)
- [Guide de Démarrage Rapide](#-guide-de-démarrage-rapide)
- [Configuration](#️-configuration)
- [Utilisation Avancée](#-utilisation-avancée)
- [Structure du Projet](#-structure-du-projet)
- [Tests](#-tests)
- [Limites et Avertissements](#️-limites-et-avertissements)
- [Contribution](#-contribution)

---

## 🎯 Vue d'ensemble

**RAG Formulaire** est une preuve de concept de système RAG (Retrieval-Augmented Generation) conçu pour interroger intelligemment les formulaires d'Immigration, Réfugiés et Citoyenneté Canada (IRCC) en français.

### Architecture monorepo (enterprise-ready)

- **Backend API** : FastAPI dans `backend/` qui enveloppe le module `rag_formulaire` existant.
- **Portail Web** : SPA React/Vite dans `frontend/web-portal` pour l'usage intranet.
- **Bot Teams** : Scaffold Bot Framework dans `frontend/teams-bot` pour Microsoft Teams.
- **Infra & Ops** : Docker, docker-compose et manifests Kubernetes dans `infra/`, scripts d'indexation dans `infra/scripts`.
- **Docs** : Architecture, sécurité et opérations dans `docs/`.

### Démarrage rapide (local)

1. **Construire les index** : `python infra/scripts/init_indexes.py`
2. **API** :
   ```bash
   cd backend
   pip install -r requirements.txt
   uvicorn api.main:app --reload
   ```
3. **Front web** :
   ```bash
   cd frontend/web-portal
   npm install
   npm run dev
   ```
   puis ouvrir http://localhost:5173 (configurer `VITE_API_BASE_URL`).
4. **Stack complète** : `cd infra && docker-compose up --build`

### Tests

- Backend : `PYTHONPATH=backend/src:src pytest backend/tests`
- Front (placeholder lint) : `npm run lint` dans `frontend/web-portal`.

### Cas d'utilisation

- 🔍 **Recherche de formulaires** : "Quel formulaire utiliser pour un permis de travail ?"
- 📝 **Questions spécifiques** : "Quelles informations sont requises dans le formulaire IMM 5476 ?"
- 🎯 **Ciblage précis** : Détection automatique et filtrage par code de formulaire (IMM/CIT)
- 🌐 **Multilingue** : Questions en français ou anglais, réponses toujours en français

### Caractéristiques Techniques

- **Recherche Hybride** : Combine BM25 (lexical) et recherche vectorielle (sémantique)
- **Reranking Cross-Encoder** : Améliore la pertinence des résultats
- **Garde-fous CRAG** : Évaluation de la qualité des preuves avant génération
- **Détection d'Hallucinations** : Validation automatique des codes de formulaire mentionnés
- **Auto-Réflexion Améliorée** : Le LLM critique ses propres réponses avec 4 points de vérification
- **LLM Local** : Mistral-7B-Instruct avec quantification 4-bit sur GPU
- **Parsing Avancé** : Docling avec support OCR pour PDF complexes
- **Singleton Pattern** : Optimisation mémoire GPU (~50% d'économie)
- **Chunks Optimisés** : 400 tokens avec 80 tokens de chevauchement pour meilleur contexte

---

## 🏗️ Architecture du Système

Le pipeline RAG est composé de plusieurs modules interconnectés, organisés en trois phases principales : **Ingestion**, **Récupération**, et **Génération**.

```mermaid
graph TB
    subgraph "📥 PHASE 1: INGESTION DES DONNÉES"
        A[🌐 Web Crawler<br/>downloader.py] -->|Formulaires PDF| B[📄 Parser Docling<br/>parser_docling.py]
        B -->|Texte + Métadonnées| C[✂️ Chunking Contextuel<br/>chunking.py]
        C -->|Chunks enrichis| D[🗂️ Indexation]
        
        subgraph D[🗂️ Indexation - indexing.py]
            D1[📊 Index BM25<br/>Recherche Lexicale]
            D2[🧠 Index Vectoriel<br/>ChromaDB + Embeddings]
        end
        
        C --> D1
        C --> D2
    end
    
    subgraph "🔍 PHASE 2: RÉCUPÉRATION & RERANKING"
        E[❓ Question Utilisateur] --> F[🌍 Handler Multilingue<br/>query_processing.py]
        F -->|Question normalisée| G{🎯 Routeur Agentique}
        
        G -->|DIRECT| H[🔎 Retrieval Hybride]
        G -->|MULTI_STEP| I[📋 Décomposition]
        I --> H
        
        subgraph H[🔎 Retrieval Hybride - retrieval.py]
            H1[🔍 Détection Code Formulaire<br/>ex: IMM 5476]
            H2[📊 Recherche BM25<br/>+ Filtrage]
            H3[🧠 Recherche Vectorielle<br/>+ Filtrage Metadata]
            H4[🔀 Fusion RRF<br/>Reciprocal Rank Fusion]
        end
        
        H --> H1
        H1 --> H2
        H1 --> H3
        H2 --> H4
        H3 --> H4
        
        H4 --> J[⚖️ Cross-Encoder Reranker<br/>reranker.py]
        J -->|Top-K chunks| K{✅ Évaluation CRAG<br/>evaluation.py}
    end
    
    subgraph "🤖 PHASE 3: GÉNÉRATION & VALIDATION"
        K -->|✅ Preuves Fortes| L[🧠 LLM Mistral-7B<br/>llm.py - Singleton]
        K -->|❌ Preuves Faibles| M[⚠️ Message de Fallback]
        
        L --> N[🔍 Auto-Réflexion<br/>evaluation.py]
        N --> O[📝 Réponse Finale<br/>+ Sources + Disclaimer]
        M --> O
    end
    
    O --> P[👤 Utilisateur]
    
    style A fill:#e1f5ff
    style B fill:#e1f5ff
    style C fill:#e1f5ff
    style D1 fill:#fff3cd
    style D2 fill:#fff3cd
    style F fill:#d4edda
    style G fill:#d4edda
    style H1 fill:#cfe2ff
    style H2 fill:#cfe2ff
    style H3 fill:#cfe2ff
    style H4 fill:#cfe2ff
    style J fill:#f8d7da
    style K fill:#f8d7da
    style L fill:#e2e3e5
    style N fill:#e2e3e5
    style O fill:#d1ecf1
    
    classDef phaseStyle stroke:#333,stroke-width:2px
    class A,B,C,D1,D2 phaseStyle
    class F,G,H1,H2,H3,H4,J,K phaseStyle
    class L,N,O phaseStyle
```

### Légende des Composants

| Icône | Composant | Description | Module |
|-------|-----------|-------------|--------|
| 🌐 | **Web Crawler** | Télécharge les formulaires IRCC depuis le site officiel | `downloader.py` |
| 📄 | **Parser Docling** | Extraction de texte avec OCR si nécessaire | `parser_docling.py` |
| ✂️ | **Chunking Contextuel** | Découpe intelligente respectant sections/questions | `chunking.py` |
| 📊 | **Index BM25** | Recherche lexicale par mots-clés | `indexing.py` |
| 🧠 | **Index Vectoriel** | Recherche sémantique via embeddings | `indexing.py` |
| 🌍 | **Handler Multilingue** | Normalisation et détection de langue | `query_processing.py` |
| 🎯 | **Routeur Agentique** | Sélection de la stratégie (DIRECT/MULTI_STEP) | `query_processing.py` |
| 🔍 | **Détection Code** | Reconnaissance automatique de codes formulaire (IMM/CIT) | `retrieval.py` |
| 🔀 | **Fusion RRF** | Combinaison des résultats BM25 + vecteurs | `retrieval.py` |
| ⚖️ | **Cross-Encoder Reranker** | Amélioration de la pertinence | `reranker.py` |
| ✅ | **Évaluation CRAG** | Validation de la qualité des preuves | `evaluation.py` |
| 🧠 | **LLM Mistral-7B** | Génération de réponses en français | `llm.py` |
| 🔍 | **Auto-Réflexion** | Vérification de cohérence de la réponse | `evaluation.py` |

---

## ✨ Fonctionnalités Principales

### 🎯 Recherche Intelligente

- **Détection Automatique de Formulaire** : Détecte les codes (IMM 5476, CIT 0002) et filtre les résultats
- **Recherche Hybride** : Combine recherche lexicale (BM25) et sémantique (vecteurs)
- **Reranking Avancé** : Cross-encoder pour améliorer la pertinence
- **Fusion RRF** : Reciprocal Rank Fusion pour combiner les scores

### 🛡️ Garde-fous et Sécurité

- **Évaluation CRAG** : Vérifie la qualité des preuves avant génération
- **Validation de Codes** : Détecte et bloque les hallucinations de codes de formulaire (IMM/CIT)
- **Auto-Réflexion Améliorée** : Critique structurée en 4 points avec détection de 8 mots-clés problématiques
- **Messages de Fallback** : Refuse de répondre si les preuves sont insuffisantes
- **Disclaimer Automatique** : Avertissement légal sur chaque réponse
- **Mode Strict Configurable** : Option de vérification n-gram ultra-stricte

### 🚀 Optimisations Performantes

- **LLM Singleton** : Une seule instance du modèle (économie mémoire ~50%)
- **Quantification 4-bit** : Chargement optimisé sur GPU T4/Tesla
- **Deduplication** : Évite les doublons dans le manifest
- **Chunking Contextuel** : Préserve le contexte avant/après chaque chunk

---

## 📦 Installation

### Prérequis

- Python 3.10 ou supérieur
- (Optionnel) GPU CUDA pour Mistral-7B-Instruct
- (Optionnel) Module `bitsandbytes` pour quantification 4-bit

### Installation Standard

```bash
# Cloner le dépôt
git clone https://github.com/abdelmajidlra/rag-formulaire.git
cd rag-formulaire

# Installer en mode développement
pip install -e .
```

### Installation pour GPU (T4/Colab)

```bash
# Installer avec support quantification
pip install -e .
pip install bitsandbytes
```

---

## 🚀 Guide de Démarrage Rapide

### 1. Construction de l'Index

```bash
python -m rag_formulaire.ingest
```

**Ce que fait cette commande :**
- Télécharge au moins 30 formulaires IRCC en français
- Parse les PDFs avec Docling (OCR si nécessaire)
- Découpe en chunks contextuels
- Construit les index BM25 et vectoriel
- Génère `data/forms_manifest.json`

**Durée estimée :** 10-15 minutes (selon la connexion)

### 2. Poser des Questions

#### Mode Interactif (CLI)

```bash
python -m rag_formulaire.cli
```

**Exemples de questions :**

```
>>> Quel formulaire utiliser pour un permis de travail ?
>>> À quoi sert le formulaire IMM 5476 ?
>>> Quelles informations sont requises pour la déclaration d'union de fait ?
```

#### Mode Programmatique

```python
from rag_formulaire.cli import answer_question

result = answer_question("Quel formulaire pour un permis de travail ?")
print(result["answer"])
print(result["sources"])
```

---

## ⚙️ Configuration

Les paramètres de configuration se trouvent dans `src/rag_formulaire/config.py`. Vous pouvez les surcharger via des variables d'environnement :

| Variable | Défaut | Description |
|----------|--------|-------------|
| `RAG_FORM_BASE_DIR` | `.` | Répertoire racine pour les données |
| `RAG_FORM_MIN_FORMS` | `40` | Nombre minimum de formulaires à télécharger |
| `RAG_FORM_MAX_SYNTH` | `0` | Nombre de formulaires synthétiques (désactivé) |
| `RAG_FORM_ENABLE_GRAPHRAG` | `false` | Activer GraphRAG (expérimental) |
| `RAG_FORM_STRICT_VERIFICATION` | `false` | Mode strict : vérification n-gram (peut bloquer réponses valides) |
| `RAG_FORM_CHUNK_SIZE` | `400` | Taille des chunks en tokens (était 200) |
| `RAG_FORM_CHUNK_OVERLAP` | `80` | Chevauchement entre chunks en tokens (était 30) |
| `GEN_MODEL_NAME` | `mistralai/Mistral-7B-Instruct-v0.2` | Modèle LLM HuggingFace |
| `GEN_LOAD_4BIT` | `True` | Quantification 4-bit sur GPU |

**Exemple de configuration personnalisée :**

```bash
export RAG_FORM_MIN_FORMS=50
export RAG_FORM_BASE_DIR=/data/rag
python -m rag_formulaire.ingest
```

---

## 🔬 Utilisation Avancée

### Notebook Colab

Un notebook optimisé pour Google Colab (GPU T4) est disponible dans `notebooks/colab-ircc-rag-poc.ipynb`.

**Fonctionnalités du notebook :**
- Configuration automatique du dépôt
- Installation des dépendances
- Construction de l'index avec quota réduit (30 formulaires)
- Tests interactifs avec affichage formaté
- Gestion mémoire GPU optimisée

### Intégration dans une Application

```python
from rag_formulaire.indexing import load_indexes
from rag_formulaire.retrieval import HybridRetriever
from rag_formulaire.llm import LocalLLM
from rag_formulaire.evaluation import CRAGEvaluator

# Charger les composants (singleton pour LLM)
index = load_indexes()
retriever = HybridRetriever(index)
llm = LocalLLM()  # Singleton - une seule instance
evaluator = CRAGEvaluator()

# Pipeline complet
def query_pipeline(question: str):
    candidates = retriever.retrieve(question)
    
    if not evaluator.is_evidence_strong([1.0] * len(candidates), candidates):
        return {"answer": evaluator.fallback_message()}
    
    evidence_text = "\n".join([c.base_chunk.content for c in candidates[:5]])
    answer = llm.chat("Réponds en français basé sur ces extraits", evidence_text)
    
    return {"answer": answer, "evidence": candidates[:5]}
```

---

## 📁 Structure du Projet

```
rag-formulaire/
├── src/
│   └── rag_formulaire/
│       ├── __init__.py
│       ├── config.py              # Configuration centralisée
│       ├── downloader.py          # 🌐 Crawler de formulaires IRCC
│       ├── parser_docling.py      # 📄 Parsing PDF avec OCR
│       ├── chunking.py            # ✂️ Découpe contextuelle
│       ├── indexing.py            # 🗂️ Index BM25 + Vecteurs
│       ├── data_models.py         # 📊 Modèles de données
│       ├── query_processing.py    # 🌍 Traitement multilingue
│       ├── retrieval.py           # 🔍 Recherche hybride + filtrage
│       ├── reranker.py            # ⚖️ Cross-encoder reranking
│       ├── evaluation.py          # ✅ CRAG + auto-réflexion
│       ├── llm.py                 # 🧠 LLM Mistral (singleton)
│       ├── graph_rag.py           # 🕸️ GraphRAG (expérimental)
│       ├── ingest.py              # 📥 Pipeline d'ingestion
│       └── cli.py                 # 💬 Interface en ligne de commande
├── notebooks/
│   └── colab-ircc-rag-poc.ipynb  # 📓 Notebook Colab optimisé
├── tests/                         # 🧪 Tests unitaires
├── data/                          # 📂 Données générées (ignoré par git)
│   ├── raw/forms/                 # PDFs téléchargés
│   ├── index/                     # Index BM25 + ChromaDB
│   └── forms_manifest.json        # Métadonnées des formulaires
├── pyproject.toml                 # 📦 Configuration du projet
└── README.md                      # 📖 Ce fichier
```

---

## 🧪 Tests

Exécutez les tests unitaires avec pytest :

```bash
# Tous les tests
pytest

# Avec couverture de code
pytest --cov=rag_formulaire --cov-report=html

# Tests spécifiques
pytest tests/test_downloader.py -v
```

**Couverture actuelle :**
- Tests de téléchargement avec formulaires factices
- Tests de parsing et chunking
- Tests d'indexation et récupération
- Tests de garde-fous CRAG

---

## ⚠️ Limites et Avertissements

### Limites Techniques

- **Dépendance Réseau** : Nécessite un accès internet pour télécharger les formulaires IRCC
- **Ressources GPU** : LLM optimisé pour GPU T4 (15GB VRAM), mode CPU disponible mais lent
- **Performance** : Le reranking cross-encoder peut être lent sur CPU (~2-3s par requête)
- **Langue** : Optimisé pour le français, support anglais limité

### Avertissement Légal

> **⚠️ IMPORTANT** : Ce système est une preuve de concept à but informatif uniquement. Les réponses générées **ne constituent pas un avis juridique** ni un conseil en immigration. Veuillez **toujours vérifier les formulaires officiels** sur le site d'IRCC et consulter un professionnel qualifié pour votre situation spécifique.

### Exactitude

- Les réponses sont basées sur le contenu des formulaires PDF téléchargés
- Les formulaires peuvent être obsolètes si IRCC les met à jour
- Le LLM peut générer des erreurs malgré les garde-fous CRAG

---

## 🤝 Contribution

Les contributions sont les bienvenues ! Veuillez suivre ces étapes :

1. **Fork** le projet
2. Créer une branche feature (`git checkout -b feature/amelioration`)
3. Commit vos changements (`git commit -m 'Ajout amelioration'`)
4. Push vers la branche (`git push origin feature/amelioration`)
5. Ouvrir une **Pull Request**

### Guidelines de Contribution

- Suivre le style de code Black (`black .`)
- Ajouter des tests pour les nouvelles fonctionnalités
- Mettre à jour la documentation si nécessaire
- S'assurer que tous les tests passent (`pytest`)

---

## 📄 Licence

Ce projet est sous licence MIT. Voir le fichier `LICENSE` pour plus de détails.

---

## 🙏 Remerciements

- **IRCC** pour la disponibilité publique des formulaires
- **Docling** pour le parsing avancé de PDF
- **ChromaDB** pour le stockage vectoriel
- **HuggingFace** pour les modèles et l'écosystème
- **Mistral AI** pour Mistral-7B-Instruct

---

<div align="center">

**Développé avec ❤️ pour faciliter l'accès à l'information sur l'immigration**

[⬆ Retour en haut](#rag-formulaire-ircc)

</div>

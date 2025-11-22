# rag-formulaire

Preuve de concept RAG en français pour interroger les formulaires IRCC (Immigration, Réfugiés et Citoyenneté Canada). Le pipeline télécharge des formulaires PDF en français, les parse via Docling (avec OCR si nécessaire), construit un index hybride (BM25 + vecteur + reranker cross-encoder) et expose une CLI permettant de poser des questions en français avec garde-fous CRAG.

## Mise en route

```bash
git clone <repo> rag-formulaire
cd rag-formulaire
pip install -e .
```

> Le LLM par défaut est **Mistral-7B-Instruct** ; si un GPU Tesla/Colab est disponible, le chargement 4 bits est activé automatiquement. En l'absence de GPU ou de bitsandbytes, un mode CPU (ou un générateur factice) est utilisé pour rester exécutable.

## Construire l'index

```bash
python -m rag_formulaire.ingest
```

Le pipeline crée `data/forms_manifest.json`, télécharge automatiquement au moins 40 formulaires IRCC en français (aucun formulaire synthétique), les parse puis construit :
- un index BM25 sérialisé dans `data/index/bm25/`
- un magasin vectoriel Chroma dans `data/index/chroma/`

## Poser des questions

```bash
python -m rag_formulaire.cli
```

Dans la boucle interactive, tapez votre question (en français ou non). Les réponses sont toujours en français et doivent mentionner les formulaires/sources utilisés. Une clause de non-responsabilité est toujours ajoutée :

> "Cette réponse est fournie à titre informatif et ne constitue pas un avis juridique ou un conseil en immigration. Veuillez vérifier les formulaires officiels et, au besoin, consulter un professionnel qualifié."

## Architecture rapide

- `rag_formulaire.downloader` : crawling des formulaires IRCC.
- `parser_docling` : parsing PDF + OCR.
- `chunking` : découpe respectant sections/questions.
- `indexing` : BM25 + vecteurs.
- `retrieval` + `reranker` : fusion RRF + rerank cross-encoder.
- `query_processing` + `llm` : expansion, décomposition, routage agentique, génération française.
- `evaluation` : garde-fous CRAG, auto-réflexion, détection d'hallucinations.
- `ingest.py` : pipeline complet de construction d'index.
- `cli.py` : REPL pour question/réponse.

## Limites

- Dépend du téléchargement des formulaires IRCC ; sans accès réseau, l'ingestion échoue explicitement.
- Le modèle LLM local est simulé si les poids ne sont pas disponibles, pour conserver un fonctionnement hors-ligne.

## Tests

```bash
pytest
```

Des tests minimaux vérifient le téléchargement/chargement de formulaires factices, la construction d'index et les garde-fous de sécurité.

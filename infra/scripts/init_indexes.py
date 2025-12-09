from pathlib import Path

from rag_formulaire import config
from rag_formulaire.ingest import ingest_pipeline


def main() -> None:
    data_dir = Path(config.DATA_DIR)
    data_dir.mkdir(parents=True, exist_ok=True)
    ingest_pipeline()


if __name__ == "__main__":
    main()

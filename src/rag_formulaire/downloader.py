from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import List

import requests
from bs4 import BeautifulSoup
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from tqdm import tqdm

from . import config
from .data_models import FormMetadata

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

INDEX_URL = "https://www.canada.ca/fr/immigration-refugies-citoyennete/services/demande/formulaires-demande-guides.html"
ALLOWED_DOMAIN = "https://www.canada.ca"


def _ensure_dirs():
    config.RAW_FORMS_DIR.mkdir(parents=True, exist_ok=True)
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)


def _save_manifest(entries: List[FormMetadata]):
    data = [
        {
            "form_code": e.form_code,
            "title_fr": e.title_fr,
            "pdf_url": e.pdf_url,
            "local_path": str(e.local_path),
            "category": e.category,
            "last_updated": e.last_updated,
        }
        for e in entries
    ]
    config.MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(config.MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _extract_pdf_links(html: str) -> List[tuple[str, str]]:
    soup = BeautifulSoup(html, "html.parser")
    links = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        text = a.get_text(" ", strip=True)
        if href.lower().endswith(".pdf") and "imm" in href.lower():
            links.append((text or href, requests.compat.urljoin(ALLOWED_DOMAIN, href)))
    return links


def _download_pdf(url: str, target: Path) -> bool:
    try:
        resp = requests.get(url, timeout=20)
        if resp.status_code == 200 and resp.content:
            target.parent.mkdir(parents=True, exist_ok=True)
            with open(target, "wb") as f:
                f.write(resp.content)
            return True
    except Exception as exc:  # noqa: BLE001
        logger.warning("Echec téléchargement %s: %s", url, exc)
    return False


def _synthetic_pdf(path: Path, form_code: str, title: str, idx: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(path), pagesize=letter)
    text = c.beginText(40, 750)
    text.textLine(f"Formulaire {form_code} - {title}")
    text.textLine("Renseignements personnels")
    text.textLine("Nom complet: __________________")
    text.textLine("Date de naissance: __________________")
    text.textLine("Objet: démonstration hors ligne du pipeline RAG.")
    text.textLine(f"Section synthétique {idx}")
    c.drawText(text)
    c.showPage()
    c.save()


def _generate_synthetic_forms(target_count: int) -> List[FormMetadata]:
    entries: List[FormMetadata] = []
    for i in range(target_count):
        code = f"IMM {5400 + i}"
        title = f"Formulaire fictif numéro {i+1}"
        filename = f"synthetic_{i+1:03d}.pdf"
        local_path = config.RAW_FORMS_DIR / filename
        _synthetic_pdf(local_path, code, title, i + 1)
        entries.append(
            FormMetadata(
                form_code=code,
                title_fr=title,
                pdf_url="synthetic",
                local_path=local_path,
                category="Synthétique",
                last_updated="N/A",
            )
        )
    return entries


def download_french_ircc_forms(min_count: int | None = None) -> List[FormMetadata]:
    _ensure_dirs()
    min_target = min_count or config.MIN_FORMS
    entries: List[FormMetadata] = []

    # Try to load existing manifest
    if config.MANIFEST_PATH.exists():
        try:
            data = json.loads(config.MANIFEST_PATH.read_text("utf-8"))
            for item in data:
                entries.append(
                    FormMetadata(
                        form_code=item.get("form_code", ""),
                        title_fr=item.get("title_fr", ""),
                        pdf_url=item.get("pdf_url", ""),
                        local_path=Path(item.get("local_path")),
                        category=item.get("category"),
                        last_updated=item.get("last_updated"),
                    )
                )
        except json.JSONDecodeError:
            entries = []

    # If we already have enough, return
    if len(entries) >= min_target:
        logger.info("Manifest déjà présent avec %s formulaires", len(entries))
        return entries

    # Attempt crawl
    try:
        response = requests.get(INDEX_URL, timeout=20)
        response.raise_for_status()
        links = _extract_pdf_links(response.text)
        logger.info("Liens PDF détectés: %s", len(links))
        for idx, (text, url) in enumerate(tqdm(links, desc="Téléchargement")):
            form_code_match = re.search(r"(IMM|CIT)\s?-?\d{3,4}", text, re.IGNORECASE)
            form_code = form_code_match.group(0).upper().replace("  ", " ") if form_code_match else f"FORM-{idx}"
            title_fr = text
            local_path = config.RAW_FORMS_DIR / f"{form_code.replace(' ', '_')}.pdf"
            if local_path.exists() and local_path.stat().st_size > 1024:
                logger.debug("Fichier déjà présent: %s", local_path)
            else:
                success = _download_pdf(url, local_path)
                if not success:
                    continue
            entries.append(
                FormMetadata(
                    form_code=form_code,
                    title_fr=title_fr,
                    pdf_url=url,
                    local_path=local_path,
                    category=None,
                    last_updated=None,
                )
            )
            if len(entries) >= min_target:
                break
    except Exception as exc:  # noqa: BLE001
        logger.warning("Echec du crawl en ligne, utilisation de données synthétiques: %s", exc)

    # If still insufficient, generate synthetic
    if len(entries) < min_target:
        remaining = min(min_target - len(entries), config.MAX_SYNTHETIC_FORMS)
        logger.info("Génération de %s formulaires synthétiques pour atteindre %s", remaining, min_target)
        entries.extend(_generate_synthetic_forms(remaining))

    _save_manifest(entries)
    return entries

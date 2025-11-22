from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import List

import requests
from bs4 import BeautifulSoup
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


def _extract_pdf_links_from_soup(soup: BeautifulSoup) -> List[tuple[str, str]]:
    links = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        text = a.get_text(" ", strip=True)
        if href.lower().endswith(".pdf") and "imm" in href.lower():
            links.append((text or href, requests.compat.urljoin(ALLOWED_DOMAIN, href)))
    return links


def _extract_form_pages(soup: BeautifulSoup) -> List[str]:
    form_pages = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if not href.startswith("http"):
            href = requests.compat.urljoin(ALLOWED_DOMAIN, href)
        if "imm" in href.lower() and href.lower().endswith(".html") and "immigration-refugies-citoyennete" in href:
            form_pages.append(href)
    return list(dict.fromkeys(form_pages))


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
        response = requests.get(INDEX_URL, timeout=30)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        pdf_links = _extract_pdf_links_from_soup(soup)
        form_pages = _extract_form_pages(soup)

        # Enrichir en parcourant les pages individuelles
        for page_url in tqdm(form_pages, desc="Exploration des pages de formulaires"):
            try:
                page_resp = requests.get(page_url, timeout=20)
                if page_resp.status_code == 200:
                    page_soup = BeautifulSoup(page_resp.text, "html.parser")
                    pdf_links.extend(_extract_pdf_links_from_soup(page_soup))
            except Exception as exc:  # noqa: BLE001
                logger.debug("Ignoré %s: %s", page_url, exc)

        seen_urls = set()
        logger.info("Liens PDF détectés après crawl: %s", len(pdf_links))
        for idx, (text, url) in enumerate(tqdm(pdf_links, desc="Téléchargement")):
            if url in seen_urls:
                continue
            seen_urls.add(url)
            form_code_match = re.search(r"(IMM|CIT)\s?-?\d{3,4}", text, re.IGNORECASE)
            form_code = form_code_match.group(0).upper().replace("  ", " ") if form_code_match else f"FORM-{idx}"
            title_fr = text
            local_path = config.RAW_FORMS_DIR / f"{form_code.replace(' ', '_')}.pdf"
            if local_path.exists() and local_path.stat().st_size > 2048:
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
        logger.warning("Echec du crawl en ligne: %s", exc)

    # Si le crawl n'atteint pas le quota minimal, on échoue explicitement
    if len(entries) < min_target:
        raise RuntimeError(
            f"Seuls {len(entries)} formulaires téléchargés. Un minimum de {min_target} formulaires français est requis."
        )

    _save_manifest(entries)
    return entries

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
    """Download PDF with validation to prevent saving HTML error pages as PDFs."""
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        resp = requests.get(url, headers=headers, timeout=20)
        if resp.status_code == 200 and resp.content:
            # Validate that downloaded content is actually a PDF
            content_start = resp.content[:1024].lower()  # Check first 1KB
            
            # Check for PDF magic bytes
            if not content_start.startswith(b'%pdf-'):
                logger.warning("Rejected %s: not a valid PDF (missing PDF header)", url)
                return False
            
            # Reject HTML error pages disguised as PDFs
            if b'<!doctype' in content_start or b'<html' in content_start:
                logger.warning("Rejected %s: HTML content (likely error page)", url)
                return False
            
            # Validate minimum size (5KB) to catch truncated/corrupt downloads
            if len(resp.content) < 5120:
                logger.warning("Rejected %s: file too small (%d bytes)", url, len(resp.content))
                return False
            
            target.parent.mkdir(parents=True, exist_ok=True)
            with open(target, "wb") as f:
                f.write(resp.content)
            logger.debug("Downloaded valid PDF: %s (%d bytes)", target.name, len(resp.content))
            return True
    except Exception as exc:  # noqa: BLE001
        logger.warning("Echec téléchargement %s: %s", url, exc)
    return False


def download_french_ircc_forms(min_count: int | None = None) -> List[FormMetadata]:
    _ensure_dirs()
    min_target = min_count or config.MIN_FORMS
    entries: List[FormMetadata] = []

    # Track uniqueness to prevent duplicates in manifest
    seen_urls = set()
    seen_codes = set()

    # Try to load existing manifest
    if config.MANIFEST_PATH.exists():
        try:
            data = json.loads(config.MANIFEST_PATH.read_text("utf-8"))
            for item in data:
                url = item.get("pdf_url", "")
                code = item.get("form_code", "")

                # Skip if we've already seen this URL or Form Code (deduplication)
                if url in seen_urls or (code and code in seen_codes):
                    continue

                entries.append(
                    FormMetadata(
                        form_code=code,
                        title_fr=item.get("title_fr", ""),
                        pdf_url=url,
                        local_path=Path(item.get("local_path")),
                        category=item.get("category"),
                        last_updated=item.get("last_updated"),
                    )
                )
                seen_urls.add(url)
                if code:
                    seen_codes.add(code)
        except json.JSONDecodeError:
            entries = []

    # If we already have enough, return
    if len(entries) >= min_target:
        logger.info("Manifest déjà présent avec %s formulaires", len(entries))
        return entries

    # Attempt crawl
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        response = requests.get(INDEX_URL, headers=headers, timeout=30)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        pdf_links = _extract_pdf_links_from_soup(soup)
        form_pages = _extract_form_pages(soup)

        # Enrichir en parcourant les pages individuelles
        for page_url in tqdm(form_pages, desc="Exploration des pages de formulaires"):
            try:
                page_resp = requests.get(page_url, headers=headers, timeout=20)
                if page_resp.status_code == 200:
                    page_soup = BeautifulSoup(page_resp.text, "html.parser")
                    pdf_links.extend(_extract_pdf_links_from_soup(page_soup))
            except Exception as exc:  # noqa: BLE001
                logger.debug("Ignoré %s: %s", page_url, exc)

        logger.info("Liens PDF détectés après crawl: %s", len(pdf_links))

        for idx, (text, url) in enumerate(tqdm(pdf_links, desc="Téléchargement")):
            # Check uniqueness against existing entries
            if url in seen_urls:
                continue

            pattern = r"(IMM|CIT)\s?-?[\s_]?(\d{3,4})"
            match = re.search(pattern, text, re.IGNORECASE)

            # Fallback URL check logic
            if not match:
                match = re.search(pattern, url, re.IGNORECASE)

            if match:
                prefix = match.group(1).upper()
                number = match.group(2)
                form_code = f"{prefix} {number}"
            else:
                form_code = f"FORM-{idx}"

            # Also check uniqueness by form code if possible
            if form_code in seen_codes:
                continue

            seen_urls.add(url)
            seen_codes.add(form_code)

            title_fr = text
            local_path = config.RAW_FORMS_DIR / f"{form_code.replace(' ', '_')}.pdf"
            if local_path.exists() and local_path.stat().st_size > 5120:
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

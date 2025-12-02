import gc
import logging
import os
import sys
import torch
from typing import List, Dict, Any

try:
    from IPython.display import display, Markdown
except ImportError:
    display = print
    Markdown = str

logger = logging.getLogger(__name__)

def force_cleanup():
    """
    Aggressive memory cleanup for Colab T4 usage.
    """
    # Explicitly delete global result variable if it exists in the caller's frame
    # (Note: accessing caller's globals is tricky, so we rely on gc.collect mostly)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.debug("GPU cache cleared.")

def display_result(result: Dict[str, Any], manifest_list: List[Dict] = None):
    """
    Displays the RAG result and evidence in a formatted Markdown way.
    """
    # 1. Header with route
    md = f"### 🦙 Réponse (Stratégie : `{result.get('route', 'UNKNOWN')}`)\n\n"

    # 2. The answer
    md += f"{result.get('answer', '')}\n\n"

    # 3. Evidence
    evidence = result.get('evidence', [])
    if evidence:
        md += "---\n#### 🔍 Sources utilisées :\n"
        
        # Map form codes to URLs
        url_map = {m['form_code']: m['pdf_url'] for m in manifest_list} if manifest_list else {}

        for i, ev in enumerate(evidence, 1):
            chunk = ev.base_chunk
            form_code = chunk.form_code
            section = chunk.section_title
            
            # Link to official PDF if available
            if form_code in url_map:
                source_link = f"[{form_code}]({url_map[form_code]})"
            else:
                source_link = f"**{form_code}**"

            # Preview text (cleaned up)
            preview = chunk.content.replace("\n", " ")[:500] + "..."

            md += f"{i}. {source_link} — *{section}* (Page {chunk.page_number})\n"
            md += f"   > <small>{preview}</small>\n"
    else:
        md += "---\n*Aucune source trouvée ou utilisée.*"

    display(Markdown(md))

def setup_colab_env(repo_url: str = "https://github.com/abdelmajidlra/rag-formulaire.git"):
    """
    Sets up the Colab environment: clones repo, installs deps.
    """
    # This is mostly for the notebook to call if it wants to abstract it away,
    # but the notebook usually does this cell-by-cell for visibility.
    # We can keep it simple here.
    pass

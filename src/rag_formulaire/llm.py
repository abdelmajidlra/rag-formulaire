from __future__ import annotations

import logging
from typing import List

try:  # pragma: no cover - heavy dependency
    from transformers import pipeline
except Exception:  # noqa: BLE001
    pipeline = None

from . import config

logger = logging.getLogger(__name__)


class LocalLLM:
    def __init__(self):
        self.generator = None
        if pipeline is not None:
            try:  # pragma: no cover - heavy dependency
                self.generator = pipeline("text-generation", model=config.GEN_MODEL_NAME, device_map="auto")
            except Exception as exc:  # noqa: BLE001
                logger.warning("LLM non disponible (%s), utilisation d'un générateur factice.", exc)
        if self.generator is None:
            logger.info("Utilisation du générateur factice interne.")

    def generate(self, prompt: str, max_new_tokens: int = 256) -> str:
        if self.generator is None:
            # simple heuristic: echo prompt ending
            return prompt.split("Réponse:")[-1].strip() or "Réponse non disponible dans ce mode hors-ligne."
        output = self.generator(prompt, max_new_tokens=max_new_tokens, do_sample=True, temperature=0.3)
        text = output[0]["generated_text"]
        return text[len(prompt) :]

    def chat(self, system_prompt: str, user_prompt: str, max_new_tokens: int = 256) -> str:
        prompt = f"{system_prompt}\nUtilisateur: {user_prompt}\nRéponse:"
        return self.generate(prompt, max_new_tokens=max_new_tokens)

    def expand_queries(self, query: str, n: int = 3) -> List[str]:
        # simplistic expansion
        variants = [query]
        synonyms = ["permis", "demande", "formulaire", "document"]
        for i in range(1, n + 1):
            variants.append(f"{query} {synonyms[i % len(synonyms)]}")
        return list(dict.fromkeys(variants))

    def decompose(self, query: str) -> List[str]:
        # naive decomposition by punctuation
        parts = [p.strip() for p in query.replace("?", ".").split(".") if p.strip()]
        return parts if parts else [query]

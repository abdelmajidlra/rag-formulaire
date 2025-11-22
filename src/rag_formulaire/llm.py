from __future__ import annotations

import logging
from typing import List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import BitsAndBytesConfig

from . import config

logger = logging.getLogger(__name__)


class LocalLLM:
    def __init__(self):
        self.model = None
        self.tokenizer = None

        # 1) Détecter le GPU ou basculer CPU
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("Initialisation LLM sur device=%s", self.device)

        # Préparer la config 4 bits si possible
        quant_config = None
        if self.device == "cuda" and config.GEN_LOAD_4BIT:
            try:  # pragma: no cover - heavy dependency
                quant_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                )
                logger.info("Quantification 4 bits activée pour %s", config.GEN_MODEL_NAME)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Impossible d'activer la quantification 4 bits: %s", exc)
                quant_config = None

        try:  # pragma: no cover - heavy dependency
            # 2) Charger le tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(config.GEN_MODEL_NAME)

            # 3) Charger le modèle de génération
            load_kwargs = {
                "device_map": "auto" if self.device == "cuda" else None,
                "torch_dtype": torch.float16 if self.device == "cuda" else torch.float32,
                "low_cpu_mem_usage": True,
            }
            if quant_config is not None:
                load_kwargs["quantization_config"] = quant_config

            self.model = AutoModelForCausalLM.from_pretrained(
                config.GEN_MODEL_NAME,
                **load_kwargs,
            )

            if self.device != "cuda":
                self.model = self.model.to(self.device)

            logger.info("LLM Mistral chargé avec succès (%s)", config.GEN_MODEL_NAME)

        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "LLM non disponible (%s), utilisation d'un générateur factice.", exc
            )
            self.model = None
            self.tokenizer = None

        if self.model is None:
            logger.info("Utilisation du générateur factice interne (CPU pur).")

    # --- Génération principale -------------------------------------------------
    def generate(self, prompt: str, max_new_tokens: int = 256) -> str:
        # Mode fallback : pas de vrai modèle
        if self.model is None or self.tokenizer is None:
            return (
                prompt.split("Réponse:")[-1].strip()
                or "Réponse non disponible dans ce mode hors-ligne."
            )

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,      # réponses plus stables pour ton RAG
                temperature=0.1,
            )

        full_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
        # On enlève le prompt pour ne garder que la "réponse"
        return full_text[len(prompt):].strip()

    # --- API "chat" compatible avec ton code existant -------------------------
    def chat(self, system_prompt: str, user_prompt: str, max_new_tokens: int = 256) -> str:
        prompt = f"{system_prompt}\nUtilisateur: {user_prompt}\nRéponse:"
        return self.generate(prompt, max_new_tokens=max_new_tokens)

    # --- Expansion et décomposition restent simples pour le moment ------------
    def expand_queries(self, query: str, n: int = 3) -> List[str]:
        # Tu pourras plus tard les faire générer par le LLM si tu veux.
        variants = [query]
        synonyms = ["permis", "demande", "formulaire", "document"]
        for i in range(1, n + 1):
            variants.append(f"{query} {synonyms[i % len(synonyms)]}")
        return list(dict.fromkeys(variants))

    def decompose(self, query: str) -> List[str]:
        parts = [p.strip() for p in query.replace("?", ".").split(".") if p.strip()]
        return parts if parts else [query]

from __future__ import annotations

import logging
import os
from typing import List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from . import config

logger = logging.getLogger(__name__)

# Global variable to store the single instance
_SHARED_LLM = None


class LocalLLM:
    def __new__(cls):
        global _SHARED_LLM
        if _SHARED_LLM is None:
            _SHARED_LLM = super(LocalLLM, cls).__new__(cls)
            _SHARED_LLM._initialized = False
        return _SHARED_LLM

    def __init__(self):
        # Prevent re-initialization if already loaded
        if getattr(self, "_initialized", False):
            return

        self._initialized = True
        self.model = None
        self.tokenizer = None
        self.model_name = None
        self._fallback_loaded = False

        # Detect GPU or fallback to CPU
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("Initialisation LLM sur device=%s", self.device)

        quant_config = None
        if self.device == "cuda" and config.GEN_LOAD_4BIT:
            try:  # pragma: no cover - heavy dependency
                quant_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                )
                logger.info("Quantification 4 bits active pour %s", config.GEN_MODEL_NAME)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Impossible d'activer la quantification 4 bits (module bitsandbytes manquant ?): %s",
                    exc,
                )
                quant_config = None

        try:  # pragma: no cover - heavy dependency
            # 1) Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(config.GEN_MODEL_NAME)
            if self.tokenizer.pad_token_id is None and getattr(self.tokenizer, "eos_token_id", None) is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            # 2) Load generation model
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

            if self.device != "cuda" and not getattr(self.model, "is_quantized", False):
                self.model = self.model.to(self.device)

            self.model_name = config.GEN_MODEL_NAME
            logger.info("LLM charge avec succes (%s)", config.GEN_MODEL_NAME)

        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Echec du chargement du modele principal (%s): %s", config.GEN_MODEL_NAME, exc
            )
            # Fallback to a lightweight CPU model
            self._load_fallback_cpu(reason="chargement principal impossible")

        if self.model is None:
            logger.warning("ATTENTION: Aucun LLM charge. Le systeme ne pourra pas generer de reponses.")

    def _format_prompt(self, system_prompt: str, user_prompt: str) -> str:
        """Build a chat-formatted prompt, using the tokenizer chat template when available."""

        if self.tokenizer and hasattr(self.tokenizer, "apply_chat_template"):
            try:
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ]
                return self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("Echec de l'application du chat template: %s", exc)

        return f"{system_prompt}\nUtilisateur: {user_prompt}\nReponse:"

    def _load_fallback_cpu(self, reason: str | None = None):
        """Load a small CPU model to avoid GPU OOM or missing deps."""
        fallback_name = os.getenv("RAG_FORM_FALLBACK_MODEL", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
        if reason:
            logger.info("Chargement du modele de repli CPU (%s) suite a: %s", fallback_name, reason)
        else:
            logger.info("Chargement du modele de repli CPU: %s", fallback_name)

        try:
            # Free GPU memory if a model is already loaded there
            if torch.cuda.is_available():
                try:
                    if self.model is not None:
                        self.model.to("cpu")
                except Exception:  # noqa: BLE001
                    pass
                torch.cuda.empty_cache()

            self.tokenizer = AutoTokenizer.from_pretrained(fallback_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                fallback_name,
                device_map="cpu",
                torch_dtype=torch.float32,
            )
            self.device = "cpu"
            self.model_name = fallback_name
            self._fallback_loaded = True
            logger.info("Modele de repli charge avec succes sur CPU.")
        except Exception as fallback_exc:  # noqa: BLE001
            logger.error(
                "Echec du chargement du modele de repli (%s): %s. Utilisation du mode hors-ligne.",
                fallback_name,
                fallback_exc,
            )
            self.model = None
            self.tokenizer = None
            self.model_name = None
            self._fallback_loaded = False

    def _run_with_current_model(self, prompt: str, max_new_tokens: int) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt")

        # Truncate input if too long to prevent overflow errors
        if inputs.input_ids.shape[1] > 2048:
            inputs.input_ids = inputs.input_ids[:, -2048:]
            inputs.attention_mask = inputs.attention_mask[:, -2048:]

        # Match the device of the model (CPU fallback or GPU)
        try:
            model_device = next(self.model.parameters()).device
        except Exception:  # noqa: BLE001
            model_device = None

        if model_device is not None:
            inputs = inputs.to(model_device)

        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,  # more stable for RAG
                temperature=0.1,
                repetition_penalty=1.2,  # avoid loops like "pour pour pour"
                pad_token_id=self.tokenizer.eos_token_id,
            )

        full_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
        # Remove the prompt to keep only the answer
        return full_text[len(prompt):].strip()

    # --- Generation principale -------------------------------------------------
    def generate(self, prompt: str, max_new_tokens: int = 256) -> str:
        # Fallback mode: no usable model
        if self.model is None or self.tokenizer is None:
            return (
                prompt.split("Reponse:")[-1].strip()
                or "Reponse non disponible dans ce mode hors-ligne."
            )

        # Aggressive cleanup before generation
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        try:
            return self._run_with_current_model(prompt, max_new_tokens)
        except RuntimeError as exc:  # pragma: no cover - depends on CUDA
            if "out of memory" not in str(exc).lower():
                raise

            logger.warning(
                "OOM durant la generation avec %s. Passage en mode CPU leger pour terminer.",
                self.model_name,
            )
            
            # Aggressive cleanup after OOM
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Switch to CPU fallback and retry with fewer tokens to be safe
            self._load_fallback_cpu(reason="CUDA OOM pendant generate()")

            if self.model is None or self.tokenizer is None:
                return "Reponse non disponible en raison d'un manque de memoire."

            try:
                reduced_tokens = max(64, max_new_tokens // 2)
                return self._run_with_current_model(prompt, reduced_tokens)
            except Exception as fallback_exc:  # noqa: BLE001
                logger.error("Echec du fallback CPU apres OOM: %s", fallback_exc)
                return "Reponse non disponible en raison d'un manque de memoire."

    # --- API "chat" compatible avec ton code existant -------------------------
    def chat(self, system_prompt: str, user_prompt: str, max_new_tokens: int = 256) -> str:
        prompt = self._format_prompt(system_prompt, user_prompt)
        return self.generate(prompt, max_new_tokens=max_new_tokens)

    # --- Expansion et decomposition restent simples pour le moment ------------
    def expand_queries(self, query: str, n: int = 3) -> List[str]:
        # You can later delegate this to the LLM if needed.
        variants = [query]
        synonyms = ["permis", "demande", "formulaire", "document"]
        for i in range(1, n + 1):
            variants.append(f"{query} {synonyms[i % len(synonyms)]}")
        return list(dict.fromkeys(variants))

    def decompose(self, query: str) -> List[str]:
        parts = [p.strip() for p in query.replace("?", ".").split(".") if p.strip()]
        return parts if parts else [query]

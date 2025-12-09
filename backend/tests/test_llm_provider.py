import importlib
import types

import torch

from rag_formulaire import config
from rag_formulaire import llm as llm_module


def _reset_shared_llm():
    llm_module._SHARED_LLM = None


def test_default_model_is_llama(monkeypatch):
    monkeypatch.delenv("RAG_FORM_GEN_MODEL", raising=False)
    importlib.reload(importlib.import_module("rag_formulaire.config"))

    assert "llama" in config.GEN_MODEL_NAME.lower()


class DummyTokenizer:
    def __init__(self):
        self.pad_token_id = None
        self.eos_token = "<eos>"
        self.eos_token_id = 42
        self.apply_chat_template_called = False
        self.last_prompt = ""

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):  # noqa: D401
        self.apply_chat_template_called = True
        return f"<chat>{messages[0]['content']}|{messages[1]['content']}"

    def __call__(self, prompt, return_tensors=None):
        self.last_prompt = prompt
        ids = torch.tensor([[1, 2, 3]])
        return DummyInputs(ids)

    def decode(self, output, skip_special_tokens=True):  # noqa: ARG002
        return f"{self.last_prompt} <answer>"

    @property
    def pad_token(self):
        return self.pad_token_id

    @pad_token.setter
    def pad_token(self, value):  # noqa: D401
        self.pad_token_id = value


class DummyInputs(dict):
    def __init__(self, ids):
        super().__init__({"input_ids": ids, "attention_mask": torch.ones_like(ids)})

    def __getattr__(self, item):
        return self[item]

    def to(self, device):  # noqa: ARG002
        return self


class DummyModel:
    def __init__(self):
        self._param = torch.nn.Parameter(torch.zeros(1))

    def parameters(self):
        yield self._param

    def to(self, device):  # noqa: ARG002
        return self

    def generate(self, **kwargs):  # noqa: ARG002
        return torch.tensor([[0, 1, 2]])


def test_chat_template_used(monkeypatch):
    _reset_shared_llm()

    tokenizer = DummyTokenizer()
    monkeypatch.setattr(llm_module, "AutoTokenizer", types.SimpleNamespace(from_pretrained=lambda name: tokenizer))
    monkeypatch.setattr(
        llm_module,
        "AutoModelForCausalLM",
        types.SimpleNamespace(from_pretrained=lambda name, **kwargs: DummyModel()),
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    llama = llm_module.LocalLLM()
    response = llama.chat("system", "user", max_new_tokens=8)

    assert tokenizer.apply_chat_template_called is True
    assert response == "<answer>"

    _reset_shared_llm()


def test_remote_backend_is_used(monkeypatch):
    _reset_shared_llm()
    monkeypatch.setenv("RAG_FORM_GEN_ENDPOINT", "http://llama:8080")

    importlib.reload(config)
    importlib.reload(llm_module)
    _reset_shared_llm()

    calls = {}

    class DummyResponse:
        def __init__(self, payload):
            self._payload = payload

        def raise_for_status(self):  # noqa: D401
            return None

        def json(self):  # noqa: D401
            return self._payload

    def fake_post(url, json=None, headers=None, timeout=None):  # noqa: A002
        calls["url"] = url
        calls["json"] = json
        calls["headers"] = headers
        calls["timeout"] = timeout
        return DummyResponse({"generated_text": "bonjour"})

    monkeypatch.setattr(llm_module.requests, "post", fake_post)

    llama = llm_module.LocalLLM()
    response = llama.chat("system", "user", max_new_tokens=4)

    assert response == "bonjour"
    assert calls["url"] == "http://llama:8080/generate"
    assert calls["json"]["parameters"]["max_new_tokens"] == 4

    _reset_shared_llm()
    monkeypatch.delenv("RAG_FORM_GEN_ENDPOINT", raising=False)
    importlib.reload(config)
    importlib.reload(llm_module)

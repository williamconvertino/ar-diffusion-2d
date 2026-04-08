"""
Model back-ends: LLaMA (causal LM) and LLaDA (masked diffusion LM).
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class _BaseModelBackend(ABC):
    """Common interface for all model backends."""

    def __init__(self, model_name: str, device: str = "auto", **kwargs):
        self.model_name  = model_name
        self.device      = device
        self._model      = None
        self._tokenizer  = None
        self._load(model_name, device, **kwargs)

    @abstractmethod
    def _load(self, model_name: str, device: str, **kwargs) -> None: ...

    @abstractmethod
    def generate(self, prompt: str, max_new_tokens: int = 512, **kwargs) -> str: ...

    @abstractmethod
    def log_prob(self, prompt: str, completion: str) -> float | None: ...


class LlamaBackend(_BaseModelBackend):
    """Causal-LM backend (LLaMA-3, Mistral, etc.)."""

    def _load(self, model_name: str, device: str, **kwargs) -> None:
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            import torch
        except ImportError as e:
            raise ImportError("Install `transformers` and `torch`.") from e

        logger.info("Loading LLaMA-family model: %s", model_name)
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=device,
            torch_dtype=torch.float16,
            **kwargs,
        )

    def generate(self, prompt: str, max_new_tokens: int = 512, **kwargs) -> str:
        import torch
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)

        # suppress common HuggingFace generation warnings
        if self._tokenizer.pad_token_id is None:
            self._tokenizer.pad_token_id = self._tokenizer.eos_token_id
        kwargs.setdefault("pad_token_id", self._tokenizer.eos_token_id)
        kwargs.pop("max_length", None)

        with torch.no_grad():
            out = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                **kwargs,
            )
        new_tokens = out[0][inputs["input_ids"].shape[-1]:]
        return self._tokenizer.decode(new_tokens, skip_special_tokens=True)

    def log_prob(self, prompt: str, completion: str) -> float | None:
        import torch
        full_text  = prompt + completion
        enc_full   = self._tokenizer(full_text, return_tensors="pt").to(self._model.device)
        enc_prompt = self._tokenizer(prompt,    return_tensors="pt")
        prompt_len = enc_prompt["input_ids"].shape[-1]

        with torch.no_grad():
            logits = self._model(**enc_full).logits[0]   # (T, V)

        log_probs = torch.log_softmax(logits, dim=-1)
        ids = enc_full["input_ids"][0]
        total = sum(
            log_probs[i - 1, ids[i]].item()
            for i in range(prompt_len, len(ids))
        )
        return total


class LladaBackend(_BaseModelBackend):
    """
    Masked-diffusion LM backend (LLaDA / MDLM family).
    Uses iterative unmasking at inference time.
    """

    def _load(self, model_name: str, device: str, **kwargs) -> None:
        try:
            from transformers import AutoTokenizer, AutoModel
            import torch
        except ImportError as e:
            raise ImportError("Install `transformers` and `torch`.") from e

        logger.info("Loading LLaDA model: %s", model_name)

        self._tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
        )

        self._model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            **kwargs,
        )

        if device == "auto":
            device = "cuda:0" if torch.cuda.is_available() else "cpu"

        self._model = self._model.to(device)
        self._model.eval()
        self._device = next(self._model.parameters()).device

    def generate(self, prompt: str, max_new_tokens: int = 512, **kwargs) -> str:
        import torch

        inputs = self._tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        with torch.no_grad():
            out = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                steps=kwargs.pop("steps", 64),
                temperature=kwargs.pop("temperature", 0.0),
                **kwargs,
            )

        new_tokens = out[0][inputs["input_ids"].shape[-1]:]
        return self._tokenizer.decode(new_tokens, skip_special_tokens=True)

    def log_prob(self, prompt: str, completion: str) -> float | None:
        return None

class DeepSeekBackend(_BaseModelBackend):
    """
    Backend for DeepSeekMath models (deepseek-ai/deepseek-math-7b-instruct etc.).

    Differences from LlamaBackend:
      - Uses bfloat16 instead of float16
      - Loads GenerationConfig from the repo (sets pad_token_id correctly)
      - No system prompt — DeepSeekMath models don't support it
    """

    def _load(self, model_name: str, device: str, **kwargs) -> None:
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
            import torch
        except ImportError as e:
            raise ImportError("Install `transformers` and `torch`.") from e

        logger.info("Loading DeepSeekMath model: %s", model_name)
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModelForCausalLM.from_pretrained(
            model_name,
            # device_map=device,
            torch_dtype=torch.bfloat16,
            **kwargs,
        ).to(device) 
        # load the repo's generation config — sets pad_token_id, eos_token_id etc.
        self._model.generation_config = GenerationConfig.from_pretrained(model_name)
        self._model.generation_config.pad_token_id = (
            self._model.generation_config.eos_token_id
        )

    def generate(self, prompt: str, max_new_tokens: int = 512, **kwargs) -> str:
        import torch
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)
        kwargs.pop("max_length", None)
        with torch.no_grad():
            out = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                **kwargs,
            )
        new_tokens = out[0][inputs["input_ids"].shape[-1]:]
        return self._tokenizer.decode(new_tokens, skip_special_tokens=True)

    def log_prob(self, prompt: str, completion: str) -> float | None:
        import torch
        full_text  = prompt + completion
        enc_full   = self._tokenizer(full_text, return_tensors="pt").to(self._model.device)
        enc_prompt = self._tokenizer(prompt,    return_tensors="pt")
        prompt_len = enc_prompt["input_ids"].shape[-1]
        with torch.no_grad():
            logits = self._model(**enc_full).logits[0]
        log_probs = torch.log_softmax(logits, dim=-1)
        ids = enc_full["input_ids"][0]
        return sum(
            log_probs[i - 1, ids[i]].item()
            for i in range(prompt_len, len(ids))
        )


class MockBackend(_BaseModelBackend):
    """Deterministic stub for smoke-testing without a GPU."""

    def _load(self, *a, **kw): pass

    def generate(self, prompt: str, **kw) -> str:
        # always returns an invalid grid so format_valid=False — expected in tests
        return ""

    def log_prob(self, *a, **kw) -> float | None:
        return -6.0


def build_backend(
    model: str | _BaseModelBackend,
    device: str = "auto",
    model_kwargs: dict | None = None,
) -> _BaseModelBackend:
    """
    Resolve a model string or pre-built backend into a backend instance.
    Auto-detects LLaMA vs LLaDA from the model name.
    """
    if isinstance(model, _BaseModelBackend):
        return model

    name_lower = model.lower()
    kw = model_kwargs or {}
    _llada_hints = ("llada", "mdlm", "diffusion", "gsai-ml", "gsai_ml")
    _deepseek_hints = ("deepseek-math", "deepseek_math")
    if any(h in name_lower for h in _llada_hints):
        return LladaBackend(model, device=device, **kw)
    if any(h in name_lower for h in _deepseek_hints):
        return DeepSeekBackend(model, device=device, **kw)
    return LlamaBackend(model, device=device, **kw)
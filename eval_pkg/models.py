"""
Model back-ends: LLaMA (causal LM) and LLaDA (masked diffusion LM).
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path

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

## LLada sampling utilities
def add_gumbel_noise(logits, temperature):
    '''
    The Gumbel max is a method for sampling categorical distributions.
    According to arXiv:2409.02908, for MDM, low-precision Gumbel Max improves perplexity score but reduces generation quality.
    Thus, we use float64.
    '''
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (- torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def get_num_transfer_tokens(mask_index, steps):
    '''
    In the reverse process, the interval [0, 1] is uniformly discretized into steps intervals.
    Furthermore, because LLaDA employs a linear noise schedule (as defined in Eq. (8)),
    the expected number of tokens transitioned at each step should be consistent.

    This function is designed to precompute the number of tokens that need to be transitioned at each step.
    '''
    mask_num = mask_index.sum(dim=1, keepdim=True)

    base = mask_num // steps
    remainder = mask_num % steps

    num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base

    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, :remainder[i]] += 1

    return num_transfer_tokens

class LladaBackend(_BaseModelBackend):
    """
    Masked-diffusion LM backend (LLaDA / MDLM family).
    Uses iterative unmasking at inference time.
    """

    def _load(self, model_name: str, device: str, **kwargs) -> None:
        try:
            from transformers import AutoTokenizer, AutoModel
            
        except ImportError as e:
            raise ImportError("Install `transformers` and `torch`.") from e

        logger.info("Loading LLaDA model: %s", model_name)

        if device == "auto":
            device = "cuda:0" if torch.cuda.is_available() else "cpu"

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

        # The LLaDA architecture theoretically supports both left-padding and right-padding. 
        # However, the sampling code implementation is simpler with left-padding.
        if self._tokenizer.padding_side != 'left':
            self._tokenizer.padding_side = 'left'

        # If the padding ID equals the mask ID, you need to modify our generate function to achieve correct inference.
        assert self._tokenizer.pad_token_id != 126336

        self._model = self._model.to(device)
        self._model.eval()
        self._device = next(self._model.parameters()).device

    @ torch.no_grad()
    def _generate(self, model, prompt, attention_mask=None, steps=128, gen_length=128, block_length=128, temperature=0.,
                cfg_scale=0., remasking='low_confidence', mask_id=126336, logits_eos_inf=False, confidence_eos_eot_inf=False,
                viz_path = None):
        '''
        Args:
            model: Mask predictor.
            prompt: A tensor of shape (1, L).
            steps: Sampling steps, less than or equal to gen_length.
            gen_length: Generated answer length.
            block_length: Block length, less than or equal to gen_length. If less than gen_length, it means using semi_autoregressive remasking.
            temperature: Categorical distribution sampling temperature.
            cfg_scale: Unsupervised classifier-free guidance scale.
            remasking: Remasking strategy. 'low_confidence' or 'random'.
            mask_id: The toke id of [MASK] is 126336.
            logits_eos_inf: Whether to set the logits of EOS token to -inf. See Appendix B.4 of LLaDA for details
            confidence_eos_eot_inf: Whether to set the confidence of EOS and EoT token to -inf. See Appendix B.4 of LLaDA for details
        '''
        import torch
        x = torch.full((prompt.shape[0], prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
        x[:, :prompt.shape[1]] = prompt.clone()

        if attention_mask is not None:
            attention_mask = torch.cat([attention_mask, torch.ones((prompt.shape[0], gen_length), dtype=attention_mask.dtype, device=model.device)], dim=-1)

        prompt_index = (x != mask_id)

        assert gen_length % block_length == 0
        num_blocks = gen_length // block_length

        assert steps % num_blocks == 0
        steps = steps // num_blocks

        print_i = 0

        for num_block in range(num_blocks):
            block_mask_index = (x[:, prompt.shape[1] + num_block * block_length: prompt.shape[1] + (num_block + 1) * block_length:] == mask_id)
            num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)
            for i in range(steps):
                mask_index = (x == mask_id)
                if cfg_scale > 0.:
                    un_x = x.clone()
                    un_x[prompt_index] = mask_id
                    x_ = torch.cat([x, un_x], dim=0)
                    if attention_mask is not None:
                        attention_mask_ = torch.cat([attention_mask, attention_mask], dim=0)
                    logits = model(x_, attention_mask=attention_mask_).logits
                    logits, un_logits = torch.chunk(logits, 2, dim=0)
                    logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
                else:
                    logits = model(x, attention_mask=attention_mask).logits

                if logits_eos_inf:
                    logits[:, :, 126081] = -torch.inf

                logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
                x0 = torch.argmax(logits_with_noise, dim=-1) # b, l
                
                if confidence_eos_eot_inf:
                    logits_with_noise[:, :, 126081] = logits[:, :, 126348] = -torch.inf

                if remasking == 'low_confidence':
                    p = F.softmax(logits, dim=-1)
                    x0_p = torch.squeeze(
                        torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1) # b, l
                elif remasking == 'random':
                    x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
                else:
                    raise NotImplementedError(remasking)

                x0_p[:, prompt.shape[1] + (num_block + 1) * block_length:] = -np.inf

                x0 = torch.where(mask_index, x0, x)
                confidence = torch.where(mask_index, x0_p, -np.inf)

                transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
                for j in range(confidence.shape[0]):
                    _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j, i])
                    transfer_index[j, select_index] = True
                x[transfer_index] = x0[transfer_index]

                if viz_path is not None:
                    print_i = print_i + 1
                    # Get generated token sequence (assuming batch_size=1)
                    generated_token_ids = x[0, prompt.shape[1]:]  # Take first sample by reducing dimension
                    formatted_output = []
                    for token_id in generated_token_ids:
                        # Decode single token and handle newlines
                        decoded_token = self._tokenizer.decode(token_id).replace("\n", " ").replace("<|eot_id|>", " ").replace("<|endoftext|>", " ")
                        
                        # Add asterisk wrapping (preserve original space positions)
                        formatted_token = f"*{decoded_token}&"
                        formatted_output.append(formatted_token)
                    # Combine final output
                    final_output = "".join(formatted_output).strip()
                    viz_path = Path(viz_path)
                    if not viz_path.parent.exists():
                        viz_path.parent.mkdir(parents=True, exist_ok=True)
                    print(f"{print_i}, {final_output}", file=open(str(viz_path), "a"))

        return x

    def generate(self, prompt: str, max_new_tokens: int = 512, **kwargs) -> str:
        import torch

        encoded_prompt = self._tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self._device) for k, v in encoded_prompt.items()}

        input_ids = encoded_prompt['input_ids'].to(self._model.device)
        attention_mask = encoded_prompt['attention_mask'].to(self._model.device)

        # with torch.no_grad():
        #     out = self._generate(
        #         self._model,
        #         **inputs,
        #         max_new_tokens=max_new_tokens,
        #         steps=kwargs.pop("steps", 64),
        #         temperature=kwargs.pop("temperature", 0.0),
        #         **kwargs,
        #     )
        with torch.no_grad():
            out = self._generate(
                self._model,
                input_ids,
                gen_length=max_new_tokens,
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
      - No system prompt  DeepSeekMath models don't support it
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
            device_map=device,
            torch_dtype=torch.bfloat16,
            **kwargs,
        )
        # load the repo's generation config  sets pad_token_id, eos_token_id etc.
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
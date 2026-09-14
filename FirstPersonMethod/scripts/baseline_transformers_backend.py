#!/usr/bin/env python3
"""Local Transformers/PEFT generation backend for baseline_transformers."""

from __future__ import annotations

import sys
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

import baseline_vllm as baseline
from peft_adapter_compat import (
    is_local_peft_adapter,
    load_peft_adapter_with_compat,
)


TOKENIZER_FILE_MARKERS = (
    "tokenizer_config.json",
    "tokenizer.json",
    "tokenizer.model",
)


def local_path_has_tokenizer(path_value: str) -> bool:
    """Check whether a local checkpoint carries a saved tokenizer."""
    path = Path(path_value)
    return path.is_dir() and any(
        (path / marker).is_file() for marker in TOKENIZER_FILE_MARKERS
    )


def choose_tokenizer_source(
    model_name_or_path: str,
    tokenizer_name_or_path: str,
    is_adapter: bool,
    base_model_name_or_path: str,
) -> str:
    """Prefer an explicit/saved adapter tokenizer, then the adapter base."""
    explicit = tokenizer_name_or_path.strip()
    if explicit:
        return explicit
    if not is_adapter or local_path_has_tokenizer(model_name_or_path):
        return model_name_or_path
    if not base_model_name_or_path:
        raise ValueError("Cannot resolve tokenizer source for PEFT adapter")
    return base_model_name_or_path


class TransformersClient(baseline.BaseLLMClient):
    """A local Hugging Face causal-LM backend with the baseline client API."""

    def __init__(
        self,
        model_name_or_path: str,
        tokenizer_name_or_path: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        do_sample: bool,
        enable_thinking: bool,
        dtype: str,
        device: str,
        device_map: Optional[str],
        load_in_4bit: bool,
        load_in_8bit: bool,
        attn_implementation: Optional[str],
        trust_remote_code: bool,
        local_files_only: bool,
        revision: str,
        seed: int,
        base_model_name_or_path: str = "",
        model_is_adapter: bool = False,
    ) -> None:
        if sys.version_info < (3, 10):
            raise RuntimeError(
                "Transformers inference requires Python >= 3.10; active Python is "
                + sys.version.split()[0]
            )
        try:
            import torch
            from transformers import (
                AutoModelForCausalLM,
                AutoTokenizer,
                BitsAndBytesConfig,
            )
        except ImportError as exc:
            raise RuntimeError(
                "Missing local-inference dependencies. Install requirements.txt "
                "(torch, transformers, accelerate, and optionally bitsandbytes)."
            ) from exc

        self.torch = torch
        self.model_name_or_path = model_name_or_path.strip()
        self.is_adapter = bool(model_is_adapter) or is_local_peft_adapter(
            self.model_name_or_path
        )
        self.base_model_name_or_path = base_model_name_or_path.strip()
        self.max_tokens = int(max_tokens)
        self.temperature = float(temperature)
        self.top_p = float(top_p)
        self.do_sample = bool(do_sample)
        self.enable_thinking = bool(enable_thinking)
        self.seed = int(seed)
        self.request_count = 0
        self.generation_lock = threading.Lock()

        if not self.model_name_or_path:
            raise ValueError("--model must not be empty")
        if self.max_tokens <= 0:
            raise ValueError("--max_tokens must be positive")
        if self.do_sample and self.temperature <= 0:
            raise ValueError("--temperature must be positive when --do_sample true")
        if not 0 < self.top_p <= 1:
            raise ValueError("--top_p must be in (0, 1]")
        if load_in_4bit and load_in_8bit:
            raise ValueError("--load_in_4bit and --load_in_8bit are mutually exclusive")

        self.device = self._resolve_device(torch, device)
        model_dtype = self._resolve_dtype(torch, dtype, self.device)
        normalized_device_map: Any = (device_map or "").strip()
        quantization_config = None
        if load_in_4bit or load_in_8bit:
            try:
                import bitsandbytes  # noqa: F401
            except ImportError as exc:
                raise RuntimeError(
                    "4/8-bit inference requires bitsandbytes: pip install bitsandbytes"
                ) from exc
            if load_in_4bit:
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=model_dtype,
                )
            else:
                quantization_config = BitsAndBytesConfig(load_in_8bit=True)
            if not normalized_device_map:
                normalized_device_map = self._single_device_map(self.device)

        common_load_kwargs: Dict[str, Any] = {
            "trust_remote_code": trust_remote_code,
            "local_files_only": local_files_only,
        }
        if revision.strip():
            common_load_kwargs["revision"] = revision.strip()

        peft_model_class = None
        peft_load_kwargs: Dict[str, Any] = {
            "local_files_only": local_files_only,
        }
        if revision.strip():
            peft_load_kwargs["revision"] = revision.strip()
        if self.is_adapter:
            try:
                from peft import PeftConfig, PeftModel
            except ImportError as exc:
                raise RuntimeError(
                    "PEFT adapter inference requires peft: pip install peft"
                ) from exc
            peft_model_class = PeftModel
            adapter_config = PeftConfig.from_pretrained(
                self.model_name_or_path,
                **peft_load_kwargs,
            )
            recorded_base = str(
                getattr(adapter_config, "base_model_name_or_path", "") or ""
            ).strip()
            if not self.base_model_name_or_path:
                self.base_model_name_or_path = recorded_base
            if not self.base_model_name_or_path:
                raise ValueError(
                    "PEFT adapter has no base_model_name_or_path; pass "
                    "--base_model explicitly"
                )

        tokenizer_source = choose_tokenizer_source(
            self.model_name_or_path,
            tokenizer_name_or_path,
            self.is_adapter,
            self.base_model_name_or_path,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_source,
            use_fast=True,
            **common_load_kwargs,
        )
        if self.tokenizer.pad_token_id is None:
            if self.tokenizer.eos_token_id is None:
                raise ValueError("Tokenizer has neither a pad token nor an EOS token")
            self.tokenizer.pad_token = self.tokenizer.eos_token
        if not getattr(self.tokenizer, "chat_template", None):
            raise ValueError(
                "Tokenizer has no chat template. Use an instruct/chat checkpoint or "
                "pass its tokenizer with --tokenizer."
            )

        model_kwargs: Dict[str, Any] = {
            "dtype": model_dtype,
            "low_cpu_mem_usage": True,
            **common_load_kwargs,
        }
        if normalized_device_map:
            model_kwargs["device_map"] = normalized_device_map
        if quantization_config is not None:
            model_kwargs["quantization_config"] = quantization_config
        if (attn_implementation or "").strip():
            model_kwargs["attn_implementation"] = attn_implementation.strip()

        model_source = (
            self.base_model_name_or_path
            if self.is_adapter
            else self.model_name_or_path
        )
        base_model = AutoModelForCausalLM.from_pretrained(
            model_source,
            **model_kwargs,
        )
        adapter_load_summary: Dict[str, Any] = {}
        if self.is_adapter:
            if peft_model_class is None:
                raise RuntimeError("Internal error: PeftModel was not imported")
            self.model, adapter_load_summary = load_peft_adapter_with_compat(
                base_model,
                self.model_name_or_path,
                peft_model_class,
                torch,
                is_trainable=False,
                **peft_load_kwargs,
            )
        else:
            self.model = base_model
        if not normalized_device_map:
            self.model.to(self.device)
        self.model.eval()
        self.input_device = self._model_input_device(self.model, self.device)

        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

        footprint = None
        if hasattr(self.model, "get_memory_footprint"):
            try:
                footprint = int(self.model.get_memory_footprint())
            except (TypeError, RuntimeError):
                footprint = None
        print(
            "[model] backend=transformers model={} dtype={} input_device={} "
            "device_map={} quantization={} adapter={} base_model={} "
            "tokenizer={} adapter_namespace_remaps={} memory_gib={}".format(
                self.model_name_or_path,
                model_dtype,
                self.input_device,
                normalized_device_map or "single-device",
                "4bit" if load_in_4bit else "8bit" if load_in_8bit else "none",
                self.is_adapter,
                self.base_model_name_or_path or "n/a",
                tokenizer_source,
                adapter_load_summary.get("namespace_remapped_keys", 0),
                "{:.2f}".format(footprint / (1024 ** 3))
                if footprint is not None
                else "unknown",
            )
        )

    @staticmethod
    def _resolve_device(torch: Any, requested: str) -> Any:
        normalized = requested.strip().lower()
        if normalized == "auto":
            return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        resolved = torch.device(requested)
        if resolved.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(f"Requested --device {requested}, but CUDA is unavailable")
        return resolved

    @staticmethod
    def _resolve_dtype(torch: Any, requested: str, device: Any) -> Any:
        if requested == "bf16":
            return torch.bfloat16
        if requested == "fp16":
            if device.type == "cpu":
                raise ValueError("--dtype fp16 is not supported for CPU inference")
            return torch.float16
        if requested == "fp32":
            return torch.float32
        if device.type == "cuda":
            if hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
                return torch.bfloat16
            return torch.float16
        return torch.float32

    @staticmethod
    def _single_device_map(device: Any) -> Dict[str, Any]:
        if device.type == "cuda":
            return {"": device.index if device.index is not None else 0}
        return {"": str(device)}

    @staticmethod
    def _model_input_device(model: Any, fallback: Any) -> Any:
        try:
            device = model.get_input_embeddings().weight.device
            if device.type != "meta":
                return device
        except (AttributeError, RuntimeError):
            pass
        for parameter in model.parameters():
            if parameter.device.type != "meta":
                return parameter.device
        return fallback

    def _generate_chat(
        self,
        system_prompt: str,
        user_prompt: str,
        assistant_prefix: Optional[str] = None,
    ) -> baseline.ChatResponse:
        messages: List[Dict[str, str]] = []
        if system_prompt.strip():
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})

        is_prefill = assistant_prefix is not None
        if is_prefill:
            if not isinstance(assistant_prefix, str) or not assistant_prefix:
                raise ValueError("assistant_prefix must be a non-empty string")
            messages.append({"role": "assistant", "content": assistant_prefix})

        with self.generation_lock:
            try:
                template_kwargs: Dict[str, Any] = {
                    "tokenize": True,
                    "add_generation_prompt": not is_prefill,
                    "return_dict": True,
                    "return_tensors": "pt",
                    "enable_thinking": self.enable_thinking,
                }
                if is_prefill:
                    # Keep the final assistant message open so generation
                    # continues the supplied prefix instead of starting a new
                    # turn or inserting an end-of-message token.
                    template_kwargs["continue_final_message"] = True
                encoded = self.tokenizer.apply_chat_template(messages, **template_kwargs)
            except TypeError as exc:
                raise RuntimeError(
                    "The tokenizer chat template is incompatible with local chat "
                    "generation{}; check the model/tokenizer pair and Transformers "
                    "version.".format(" with assistant prefill" if is_prefill else "")
                ) from exc

            if isinstance(encoded, self.torch.Tensor):
                model_inputs: Dict[str, Any] = {
                    "input_ids": encoded,
                    "attention_mask": self.torch.ones_like(encoded),
                }
            else:
                model_inputs = dict(encoded)
            model_inputs = {
                key: value.to(self.input_device) if hasattr(value, "to") else value
                for key, value in model_inputs.items()
            }
            input_length = int(model_inputs["input_ids"].shape[-1])
            generation_kwargs: Dict[str, Any] = {
                "max_new_tokens": self.max_tokens,
                "do_sample": self.do_sample,
                "pad_token_id": self.tokenizer.pad_token_id,
                "use_cache": True,
            }
            if self.do_sample:
                generation_kwargs["temperature"] = self.temperature
                generation_kwargs["top_p"] = self.top_p

            self.torch.manual_seed(self.seed + self.request_count)
            if self.torch.cuda.is_available():
                self.torch.cuda.manual_seed_all(self.seed + self.request_count)
            self.request_count += 1
            with self.torch.inference_mode():
                output_ids = self.model.generate(
                    **model_inputs,
                    **generation_kwargs,
                )
            generated_ids = output_ids[0, input_length:]
            content = self.tokenizer.decode(
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            ).strip()

        if not content:
            raise RuntimeError("Empty content from local Transformers generation")
        raw = {
            "backend": "transformers",
            "model": self.model_name_or_path,
            "adapter": self.is_adapter,
            "base_model": self.base_model_name_or_path or None,
            "input_tokens": input_length,
            "output_tokens": int(generated_ids.shape[-1]),
            "do_sample": self.do_sample,
            "temperature": self.temperature if self.do_sample else None,
            "top_p": self.top_p if self.do_sample else None,
            "enable_thinking": self.enable_thinking,
            "assistant_prefill": is_prefill,
        }
        return baseline.ChatResponse(content=content, raw=raw)

    def chat(self, system_prompt: str, user_prompt: str) -> baseline.ChatResponse:
        return self._generate_chat(system_prompt, user_prompt)

    def continue_chat(
        self,
        system_prompt: str,
        user_prompt: str,
        assistant_prefix: str,
    ) -> baseline.ChatResponse:
        """Continue an unfinished assistant response from an exact text prefix."""
        return self._generate_chat(system_prompt, user_prompt, assistant_prefix)

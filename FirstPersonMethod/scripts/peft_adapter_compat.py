#!/usr/bin/env python3
"""Strict PEFT adapter loading with Qwen3.5 wrapper compatibility.

Some Qwen3.5 training paths instantiate the conditional-generation wrapper and
save language LoRA tensors below ``model.language_model``.  Text-only
``AutoModelForCausalLM`` inference/training instantiates the same backbone below
``model`` instead.  PEFT otherwise initializes fresh LoRA tensors, warns about
missing keys, and continues with an ineffective adapter.

This module detects that local checkpoint namespace and supplies PEFT's
in-memory ``key_mapping``.  Checkpoint files are never rewritten.  Missing
adapter warnings are promoted to errors so callers cannot silently continue
from the base model.
"""

from __future__ import annotations

import inspect
import warnings
from pathlib import Path
from typing import Any, Dict, Optional


PEFT_STATE_PREFIX = "base_model.model."
PEFT_SAFETENSORS_NAME = "adapter_model.safetensors"
PEFT_BIN_NAME = "adapter_model.bin"


def is_local_peft_adapter(path_value: str) -> bool:
    """Return whether a local directory contains a PEFT adapter config."""
    path = Path(path_value)
    return path.is_dir() and (path / "adapter_config.json").is_file()


def _remove_language_model_wrapper_from_peft_key(key: str) -> str:
    """Map a VLM-wrapped PEFT key to the equivalent text-only model key.

    PEFT applies ``key_mapping`` after removing its ``base_model.model.``
    prefix, so this helper deliberately operates on the prefix-free form.
    Qwen3.5 checkpoints may contain either ``model.language_model.layers`` or
    ``language_model.model.layers`` depending on the training model class.
    """
    if key.startswith("language_model."):
        return key[len("language_model.") :]
    return key.replace(".language_model.", ".", 1)


def _local_adapter_state_keys(adapter_path: str, torch: Any) -> list[str]:
    """Read local adapter keys without materializing safetensors values."""
    adapter_dir = Path(adapter_path)
    safetensors_path = adapter_dir / PEFT_SAFETENSORS_NAME
    if safetensors_path.is_file():
        try:
            from safetensors import safe_open
        except ImportError as exc:
            raise RuntimeError(
                "Inspecting a local PEFT safetensors checkpoint requires "
                "safetensors; install the project requirements"
            ) from exc
        with safe_open(str(safetensors_path), framework="pt", device="cpu") as handle:
            return list(handle.keys())

    bin_path = adapter_dir / PEFT_BIN_NAME
    if bin_path.is_file():
        try:
            state_dict = torch.load(
                str(bin_path), map_location="cpu", weights_only=True
            )
        except TypeError:
            # ``weights_only`` is unavailable on older supported PyTorch builds.
            state_dict = torch.load(str(bin_path), map_location="cpu")
        if not isinstance(state_dict, dict):
            raise ValueError(f"{bin_path} does not contain a state dictionary")
        return list(state_dict)

    return []


def build_local_adapter_key_mapping(
    adapter_path: str, torch: Any
) -> tuple[Optional[Dict[str, str]], int]:
    """Build PEFT's exact VLM-wrapper-to-text key mapping for a local adapter."""
    if not is_local_peft_adapter(adapter_path):
        return None, 0

    mapping: Dict[str, str] = {}
    for stored_key in _local_adapter_state_keys(adapter_path, torch):
        if ".language_model." not in stored_key and not stored_key.startswith(
            "language_model."
        ):
            continue
        prefix_free_key = stored_key.removeprefix(PEFT_STATE_PREFIX)
        mapped_key = _remove_language_model_wrapper_from_peft_key(prefix_free_key)
        if mapped_key != prefix_free_key:
            mapping[prefix_free_key] = mapped_key
    return (mapping or None), len(mapping)


def _assert_nonzero_loaded_lora(model: Any) -> None:
    """Fail if a remapped LoRA remained at PEFT's zero-output initialization."""
    lora_b_parameters = 0
    for name, parameter in model.named_parameters():
        if ".lora_B." not in name:
            continue
        lora_b_parameters += 1
        if bool(parameter.detach().count_nonzero().item()):
            return
    if not lora_b_parameters:
        raise RuntimeError(
            "The adapter namespace was remapped, but the loaded model has no "
            "LoRA-B parameters"
        )
    raise RuntimeError(
        "The adapter namespace was remapped, but every LoRA-B parameter is still "
        "zero. The trained adapter weights were not applied."
    )


def load_peft_adapter_with_compat(
    base_model: Any,
    adapter_path: str,
    peft_model_class: Any,
    torch: Any,
    *,
    is_trainable: bool,
    **load_kwargs: Any,
) -> tuple[Any, dict[str, Any]]:
    """Load an adapter strictly, remapping wrapped Qwen3.5 language keys.

    Returns the loaded model and serializable compatibility metadata.  Any PEFT
    missing-adapter warning becomes a ``RuntimeError`` because continuing would
    silently use freshly initialized LoRA weights.
    """
    key_mapping, remapped_keys = build_local_adapter_key_mapping(
        adapter_path, torch
    )
    effective_load_kwargs = dict(load_kwargs)
    if key_mapping:
        if "key_mapping" not in inspect.signature(
            peft_model_class.from_pretrained
        ).parameters:
            raise RuntimeError(
                "This Qwen3.5 adapter uses a .language_model. namespace, but "
                "the installed PEFT version does not support key_mapping. "
                "Upgrade PEFT to the version pinned by requirements.txt."
            )
        checkpoint_mapping = dict(
            getattr(base_model, "_checkpoint_conversion_mapping", {}) or {}
        )
        checkpoint_mapping.update(key_mapping)
        effective_load_kwargs["key_mapping"] = checkpoint_mapping
        print(
            "[adapter] detected wrapped language-model namespace; "
            f"automatically remapping {remapped_keys} checkpoint keys for "
            "text-only model loading"
        )

    with warnings.catch_warnings(record=True) as adapter_warnings:
        warnings.simplefilter("always")
        model = peft_model_class.from_pretrained(
            base_model,
            adapter_path,
            is_trainable=is_trainable,
            **effective_load_kwargs,
        )

    missing_adapter_messages = [
        str(item.message)
        for item in adapter_warnings
        if "Found missing adapter keys while loading the checkpoint"
        in str(item.message)
    ]
    for item in adapter_warnings:
        if str(item.message) not in missing_adapter_messages:
            warnings.warn(str(item.message), item.category, stacklevel=2)
    if missing_adapter_messages:
        preview = missing_adapter_messages[0]
        if len(preview) > 1200:
            preview = preview[:1200] + "..."
        raise RuntimeError(
            "PEFT did not load all adapter weights; refusing to continue with a "
            f"silently ineffective adapter. {preview}"
        )

    verified_nonzero_lora = False
    if remapped_keys:
        _assert_nonzero_loaded_lora(model)
        verified_nonzero_lora = True
        print(
            "[adapter] namespace remap verified: at least one trained LoRA-B "
            "tensor is non-zero"
        )

    return model, {
        "namespace_remapped": bool(remapped_keys),
        "namespace_remapped_keys": remapped_keys,
        "verified_nonzero_lora": verified_nonzero_lora,
    }

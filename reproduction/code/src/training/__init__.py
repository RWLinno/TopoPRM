"""Shared checkpoint loading and export for the paper training stages."""
from __future__ import annotations

import copy
import re
from pathlib import Path


def load_and_merge_adapter(model, adapter_path):
    """Load every LoRA tensor, translating a Swift text-model namespace if needed.

    Never silently retain randomly initialized LoRA tensors or discard unmatched
    saved tensors. The original checkpoint remains unchanged.
    """
    from peft import PeftConfig, get_peft_model
    from peft.utils.save_and_load import (
        get_peft_model_state_dict, load_peft_weights, set_peft_model_state_dict,
    )

    path = Path(adapter_path)
    if not path.is_dir():
        raise FileNotFoundError(f"Adapter directory not found: {path}")
    config = PeftConfig.from_pretrained(path)
    if config.peft_type != "LORA":
        raise ValueError("The paper checkpoint loader requires a LoRA adapter")
    config = copy.deepcopy(config)
    targets = config.target_modules
    module_names = set(dict(model.named_modules()))
    if isinstance(targets, str) and "language_model" in targets:
        # Swift can save text-only LoRA with the multimodal wrapper's prefix.
        if not any(re.fullmatch(targets, name) for name in module_names):
            candidate = targets.replace(r"model\.language_model(?=\.)", "model")
            candidate = candidate.replace(r"language_model\.", "").replace("language_model.", "")
            if not any(re.fullmatch(candidate, name) for name in module_names):
                raise ValueError("Adapter target_modules does not match the supplied model")
            config.target_modules = candidate
    elif isinstance(targets, (list, set)):
        # Leaf-only names need no normalization. Preserve an existing native path.
        config.target_modules = {
            target if target in module_names or "language_model." not in target
            else target.replace("language_model.", "") for target in targets
        }
    config.inference_mode = True
    weights = load_peft_weights(str(path), device="cpu")
    if not weights:
        raise ValueError("Adapter contains no tensors")
    wrapped = get_peft_model(model, config)
    expected = get_peft_model_state_dict(wrapped, save_embedding_layers=False)
    if set(weights) != set(expected):
        normalized = {key.replace(".language_model.", "."): value for key, value in weights.items()}
        if len(normalized) == len(weights) and set(normalized) == set(expected):
            weights = normalized
        else:
            raise ValueError(
                "Adapter tensors do not match the model: "
                f"missing={len(set(expected) - set(weights))}, "
                f"unexpected={len(set(weights) - set(expected))}"
            )
    for name, tensor in weights.items():
        if tensor.shape != expected[name].shape:
            raise ValueError(f"Adapter tensor shape mismatch: {name}")
        if not tensor.isfinite().all():
            raise ValueError(f"Adapter tensor is non-finite: {name}")
    result = set_peft_model_state_dict(wrapped, weights)
    missing = [name for name in result.missing_keys if "lora_" in name or "modules_to_save" in name]
    if missing or result.unexpected_keys:
        raise ValueError("PEFT did not load the complete adapter")
    return wrapped.merge_and_unload(safe_merge=True)


def save_merged_checkpoint(model, tokenizer, destination):
    """Export a standalone final checkpoint containing every preceding stage."""
    path = Path(destination)
    if path.exists():
        raise FileExistsError(f"Final checkpoint destination already exists: {path}")
    path.mkdir(parents=True)
    if hasattr(model, "merge_and_unload"):
        model = model.merge_and_unload(safe_merge=True)
    model.save_pretrained(path, safe_serialization=True)
    tokenizer.save_pretrained(path)

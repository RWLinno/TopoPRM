"""Frozen step embeddings with a lightweight directed-edge classifier."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional, Sequence

import torch
from torch import nn


DEFAULT_INSTRUCTION = (
    "Identify semantic support dependencies between mathematical reasoning steps."
)
PAIR_INSTRUCTION = (
    "Encode whether Step A directly supports Step B, Step B directly supports Step A, or neither."
)
PAIR_PROMPT_VERSION = "topoprm-pair-v1"


def _resolve_device(device: str) -> str:
    if device != "auto":
        return device
    if not torch.cuda.is_available():
        return "cpu"
    return f"cuda:{torch.cuda.current_device()}"


def format_pair_prompt(left: str, right: str) -> str:
    return (
        "Classify the direct semantic support relation between two mathematical reasoning steps.\n"
        "Label 0: neither step directly supports the other.\n"
        "Label 1: Step A directly supports Step B.\n"
        "Label 2: Step B directly supports Step A.\n"
        f"Step A: {left}\n"
        f"Step B: {right}\n"
        "Relation label:"
    )


def build_pair_features(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Build ordered pair features without an explicit position feature."""
    return torch.cat((left, right, left - right, left * right), dim=-1)


class PairDirectionHead(nn.Module):
    """Three-way classifier: no edge, left-to-right, right-to-left."""

    def __init__(self, input_dim: int = 1024, hidden_dim: int = 512, dropout: float = 0.1) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 3),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)


def _prepare_transformers(*, disable_cuda_kernels: bool = False) -> Any:
    """Disable broken optional integrations in mixed shared environments."""
    import transformers

    unavailable = lambda *args, **kwargs: False
    optional_checks = ["is_torchao_available", "is_torchaudio_available"]
    if disable_cuda_kernels:
        optional_checks.extend(("is_causal_conv1d_available", "is_flash_linear_attention_available"))
    for name in optional_checks:
        if hasattr(transformers.utils, name):
            setattr(transformers.utils, name, unavailable)
    try:
        import transformers.utils.import_utils as import_utils

        for name in optional_checks:
            if hasattr(import_utils, name):
                setattr(import_utils, name, unavailable)
    except Exception:
        pass
    try:
        import transformers.integrations.deepspeed as deepspeed_integration

        deepspeed_integration.is_deepspeed_available = unavailable
    except Exception:
        pass
    try:
        import peft.import_utils as peft_import_utils
        import peft.tuners.lora.torchao as peft_torchao

        peft_import_utils.is_torchao_available = unavailable
        peft_torchao.is_torchao_available = unavailable
    except Exception:
        pass
    return transformers


def _load_sequence_classifier(backbone: str, model_kwargs: dict[str, Any]) -> Any:
    """Load Qwen3.5 classifiers across Transformers auto-mapping versions."""
    from transformers import AutoModelForSequenceClassification

    try:
        return AutoModelForSequenceClassification.from_pretrained(backbone, **model_kwargs)
    except ValueError as exc:
        if "Qwen3_5Config" not in str(exc):
            raise
        from transformers.modeling_layers import GenericForSequenceClassification
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5PreTrainedModel

        class Qwen3_5ForSequenceClassification(
            GenericForSequenceClassification,
            Qwen3_5PreTrainedModel,
        ):
            def __init__(self, config: Any) -> None:
                # Transformers 5.2's generic classifier predates composite
                # configs and reads these fields from the top-level object.
                config.hidden_size = config.get_text_config().hidden_size
                config.pad_token_id = config.get_text_config().pad_token_id
                super().__init__(config)

        return Qwen3_5ForSequenceClassification.from_pretrained(backbone, **model_kwargs)


def _force_qwen35_torch_kernels(model: nn.Module) -> int:
    """Use Transformers' device-agnostic Qwen3.5 kernels for CPU inference."""
    try:
        from transformers.models.qwen3_5.modeling_qwen3_5 import (
            Qwen3_5GatedDeltaNet,
            torch_causal_conv1d_update,
            torch_chunk_gated_delta_rule,
            torch_recurrent_gated_delta_rule,
        )
    except (ImportError, AttributeError):
        return 0

    patched = 0
    for module in model.modules():
        if not isinstance(module, Qwen3_5GatedDeltaNet):
            continue
        module.causal_conv1d_fn = None
        module.causal_conv1d_update = torch_causal_conv1d_update
        module.chunk_gated_delta_rule = torch_chunk_gated_delta_rule
        module.recurrent_gated_delta_rule = torch_recurrent_gated_delta_rule
        patched += 1
    return patched


class FrozenStepEmbedder:
    """Lazy Qwen embedding wrapper using normalized last-token pooling."""

    def __init__(
        self,
        model_path: str,
        device: str = "auto",
        batch_size: int = 64,
        max_length: int = 256,
        instruction: str = DEFAULT_INSTRUCTION,
        dtype: Optional[str] = None,
    ) -> None:
        self.model_path = model_path
        self.device_name = device
        self.batch_size = max(1, int(batch_size))
        self.max_length = max(16, int(max_length))
        self.instruction = instruction.strip()
        self.dtype_name = dtype
        self.tokenizer: Any = None
        self.model: Any = None
        self.device: Optional[torch.device] = None

    def _load(self) -> None:
        if self.model is not None:
            return
        resolved = _resolve_device(self.device_name)
        self.device = torch.device(resolved)
        transformers = _prepare_transformers(disable_cuda_kernels=self.device.type == "cpu")
        from transformers import AutoModel, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            padding_side="left",
            trust_remote_code=True,
        )
        if self.dtype_name is None:
            dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        else:
            try:
                dtype = getattr(torch, self.dtype_name)
            except AttributeError as exc:
                raise ValueError(f"Unsupported embedding dtype: {self.dtype_name}") from exc
            if not isinstance(dtype, torch.dtype):
                raise ValueError(f"Unsupported embedding dtype: {self.dtype_name}")
        kwargs: dict[str, Any] = {"trust_remote_code": True}
        major_version = int(str(transformers.__version__).split(".", 1)[0])
        kwargs["dtype" if major_version >= 5 else "torch_dtype"] = dtype
        self.model = AutoModel.from_pretrained(self.model_path, **kwargs)
        if self.device.type == "cpu":
            _force_qwen35_torch_kernels(self.model)
        self.model.to(self.device).eval()

    def _format(self, text: str) -> str:
        if not self.instruction:
            return text
        return f"Instruct: {self.instruction}\nStep: {text}"

    def encode(self, texts: Sequence[str]) -> torch.Tensor:
        self._load()
        if not texts:
            return torch.empty((0, 0), dtype=torch.float32)
        assert self.device is not None
        chunks: list[torch.Tensor] = []
        with torch.inference_mode():
            for start in range(0, len(texts), self.batch_size):
                batch_text = [self._format(text) for text in texts[start : start + self.batch_size]]
                inputs = self.tokenizer(
                    batch_text,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                ).to(self.device)
                hidden = self.model(**inputs).last_hidden_state
                if bool(inputs["attention_mask"][:, -1].all()):
                    pooled = hidden[:, -1]
                else:
                    last = inputs["attention_mask"].sum(dim=1) - 1
                    pooled = hidden[
                        torch.arange(hidden.shape[0], device=hidden.device),
                        last,
                    ]
                chunks.append(torch.nn.functional.normalize(pooled.float(), p=2, dim=1).cpu())
        return torch.cat(chunks, dim=0)


class FrozenPairEdgeEncoder:
    """Batch semantic-edge predictor backed by a frozen embedding model."""

    def __init__(
        self,
        checkpoint_dir: str | Path,
        model_path: Optional[str] = None,
        device: str = "auto",
        batch_size: int = 64,
    ) -> None:
        from safetensors.torch import load_file

        checkpoint = Path(checkpoint_dir)
        with (checkpoint / "config.json").open(encoding="utf-8") as handle:
            self.config = json.load(handle)
        self.embedding_dim = int(self.config.get("embedding_dim", 1024))
        self.feature_mode = str(self.config.get("feature_mode", "joint_pair"))
        input_dim = self.embedding_dim if self.feature_mode == "joint_pair" else self.embedding_dim * 4
        self.head = PairDirectionHead(
            input_dim=input_dim,
            hidden_dim=int(self.config.get("hidden_dim", 512)),
            dropout=0.0,
        )
        self.head.load_state_dict(load_file(str(checkpoint / "classifier.safetensors")))
        resolved = _resolve_device(device)
        self.device = torch.device(resolved)
        self.head.to(self.device).eval()
        self.embedder = FrozenStepEmbedder(
            model_path=model_path or self.config["backbone_model"],
            device=resolved,
            batch_size=batch_size,
            max_length=int(self.config.get("max_length", 256)),
            instruction=str(self.config.get("instruction", DEFAULT_INSTRUCTION)),
        )
        self.edge_threshold = float(self.config.get("edge_threshold", 0.60))
        self.direction_margin = float(self.config.get("direction_margin", 0.15))

    def predict_batches(
        self,
        step_batches: Sequence[Sequence[str]],
        subquestion_batches: Optional[Sequence[Sequence[Optional[int]]]] = None,
    ) -> list[list[dict[str, float | int | str]]]:
        outputs: list[list[dict[str, float | int | str]]] = [[] for _ in step_batches]
        owners: list[tuple[int, int, int]] = []
        pair_texts: list[str] = []
        for batch_index, steps in enumerate(step_batches):
            subquestions = (
                subquestion_batches[batch_index]
                if subquestion_batches is not None and batch_index < len(subquestion_batches)
                else [None] * len(steps)
            )
            for left in range(len(steps)):
                for right in range(left + 1, len(steps)):
                    if (
                        left < len(subquestions)
                        and right < len(subquestions)
                        and subquestions[left] is not None
                        and subquestions[right] is not None
                        and subquestions[left] != subquestions[right]
                    ):
                        continue
                    owners.append((batch_index, left, right))
                    pair_texts.append(f"Step A: {steps[left]}\nStep B: {steps[right]}")
        if not pair_texts:
            return outputs
        if self.feature_mode == "joint_pair":
            feature_tensor = self.embedder.encode(pair_texts)
        else:
            unique_texts = list(dict.fromkeys(text for steps in step_batches for text in steps))
            embeddings = self.embedder.encode(unique_texts)
            by_text = {text: embeddings[index] for index, text in enumerate(unique_texts)}
            feature_tensor = torch.stack(
                [
                    build_pair_features(by_text[step_batches[batch][left]], by_text[step_batches[batch][right]])
                    for batch, left, right in owners
                ]
            )
        probabilities: list[torch.Tensor] = []
        with torch.inference_mode():
            for start in range(0, len(feature_tensor), 1024):
                logits = self.head(feature_tensor[start : start + 1024].to(self.device))
                probabilities.extend(torch.softmax(logits.float(), dim=-1).cpu())

        for (batch_index, left, right), probs in zip(owners, probabilities):
            label = int(torch.argmax(probs).item())
            confidence = float(probs[label].item())
            margin = abs(float(probs[1].item()) - float(probs[2].item()))
            if label == 0 or confidence < self.edge_threshold or margin < self.direction_margin:
                continue
            source, target = (left, right) if label == 1 else (right, left)
            outputs[batch_index].append(
                {
                    "source": source,
                    "target": target,
                    "confidence": confidence,
                    "direction_margin": margin,
                    "dep_type": "encoder_semantic",
                }
            )
        return outputs


class LoraPairEdgeEncoder:
    """Qwen3.5 sequence classifier adapted with a small LoRA checkpoint."""

    def __init__(
        self,
        checkpoint_dir: str | Path,
        model_path: Optional[str] = None,
        device: str = "auto",
        batch_size: int = 64,
    ) -> None:
        resolved = _resolve_device(device)
        self.device = torch.device(resolved)
        transformers = _prepare_transformers(disable_cuda_kernels=self.device.type == "cpu")
        from peft import PeftModel
        from transformers import AutoTokenizer

        checkpoint = Path(checkpoint_dir)
        with (checkpoint / "edge_encoder_config.json").open(encoding="utf-8") as handle:
            self.config = json.load(handle)
        self.batch_size = max(1, int(batch_size))
        self.max_length = int(self.config.get("max_length", 384))
        backbone = model_path or self.config["backbone_model"]
        self.tokenizer = AutoTokenizer.from_pretrained(backbone, trust_remote_code=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"
        dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32
        kwargs: dict[str, Any] = {
            "num_labels": 3,
            "trust_remote_code": True,
        }
        major_version = int(str(transformers.__version__).split(".", 1)[0])
        kwargs["dtype" if major_version >= 5 else "torch_dtype"] = dtype
        base = _load_sequence_classifier(backbone, kwargs)
        if self.device.type == "cpu":
            _force_qwen35_torch_kernels(base)
        if hasattr(base.config, "text_config"):
            base.config.text_config.pad_token_id = self.tokenizer.pad_token_id
        base.config.pad_token_id = self.tokenizer.pad_token_id
        self.model = PeftModel.from_pretrained(base, str(checkpoint)).to(self.device).eval()
        self.edge_threshold = float(self.config.get("edge_threshold", 0.60))
        self.direction_margin = float(self.config.get("direction_margin", 0.15))

    def predict_batches(
        self,
        step_batches: Sequence[Sequence[str]],
        subquestion_batches: Optional[Sequence[Sequence[Optional[int]]]] = None,
    ) -> list[list[dict[str, float | int | str]]]:
        outputs: list[list[dict[str, float | int | str]]] = [[] for _ in step_batches]
        owners: list[tuple[int, int, int]] = []
        pair_texts: list[str] = []
        for batch_index, steps in enumerate(step_batches):
            subquestions = (
                subquestion_batches[batch_index]
                if subquestion_batches is not None and batch_index < len(subquestion_batches)
                else [None] * len(steps)
            )
            for left in range(len(steps)):
                for right in range(left + 1, len(steps)):
                    if (
                        left < len(subquestions)
                        and right < len(subquestions)
                        and subquestions[left] is not None
                        and subquestions[right] is not None
                        and subquestions[left] != subquestions[right]
                    ):
                        continue
                    owners.append((batch_index, left, right))
                    pair_texts.append(format_pair_prompt(steps[left], steps[right]))
        if not pair_texts:
            return outputs

        probabilities: list[torch.Tensor] = []
        with torch.inference_mode():
            for start in range(0, len(pair_texts), self.batch_size):
                inputs = self.tokenizer(
                    pair_texts[start : start + self.batch_size],
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                ).to(self.device)
                logits = self.model(**inputs).logits
                probabilities.extend(torch.softmax(logits.float(), dim=-1).cpu())
        for (batch_index, left, right), probs in zip(owners, probabilities):
            label = int(torch.argmax(probs).item())
            confidence = float(probs[label].item())
            margin = abs(float(probs[1].item()) - float(probs[2].item()))
            if label == 0 or confidence < self.edge_threshold or margin < self.direction_margin:
                continue
            source, target = (left, right) if label == 1 else (right, left)
            outputs[batch_index].append(
                {
                    "source": source,
                    "target": target,
                    "confidence": confidence,
                    "direction_margin": margin,
                    "dep_type": "encoder_semantic",
                }
            )
        return outputs


def load_edge_encoder(
    checkpoint_dir: str | Path,
    model_path: Optional[str] = None,
    device: str = "auto",
    batch_size: int = 64,
) -> FrozenPairEdgeEncoder | LoraPairEdgeEncoder:
    checkpoint = Path(checkpoint_dir)
    if (checkpoint / "edge_encoder_config.json").is_file():
        return LoraPairEdgeEncoder(checkpoint, model_path, device, batch_size)
    return FrozenPairEdgeEncoder(checkpoint, model_path, device, batch_size)

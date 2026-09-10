"""Adapt a compact directed-edge encoder from offline semantic-teacher graphs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Optional, Sequence

import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.dag.edge_encoder import (  # noqa: E402
    PAIR_PROMPT_VERSION,
    _load_sequence_classifier,
    _prepare_transformers,
    format_pair_prompt,
)
from src.data.build_dag import segmentation_issue_reasons  # noqa: E402


LABEL_NAMES = ("no_edge", "left_to_right", "right_to_left")


def _record_split(record_id: str) -> str:
    bucket = int.from_bytes(hashlib.sha256(record_id.encode()).digest()[:4], "big") % 5
    return "validation" if bucket == 0 else "train"


def _orders(num_steps: int) -> list[list[int]]:
    candidates = [
        list(range(num_steps)),
        list(reversed(range(num_steps))),
        list(range(0, num_steps, 2)) + list(range(1, num_steps, 2)),
    ]
    seen: set[tuple[int, ...]] = set()
    output: list[list[int]] = []
    for order in candidates:
        key = tuple(order)
        if key not in seen:
            seen.add(key)
            output.append(order)
    return output


def _load_teacher_records(paths: Sequence[Path], field: str) -> list[dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                dag = row.get(field)
                if not isinstance(dag, dict):
                    continue
                nodes = sorted(dag.get("nodes", []), key=lambda node: int(node["step_id"]))
                llm_edges = [
                    edge
                    for edge in dag.get("raw_edges", [])
                    if edge.get("source_kind") == "llm"
                ]
                # The current cache cannot distinguish a true all-negative graph
                # from an LLM parsing fallback, so do not train on either.
                if len(nodes) < 2 or not llm_edges:
                    continue
                fallback_id = f"{path.name}:{line_number}"
                record_id = str(row.get("record_id", row.get("id", fallback_id)))
                by_id[record_id] = {
                    "record_id": record_id,
                    "nodes": nodes,
                    "edges": llm_edges,
                }
    return list(by_id.values())


def _malformed_node_reasons(record: dict[str, Any]) -> list[str]:
    return segmentation_issue_reasons(
        [str(node.get("raw_text", "")) for node in record["nodes"]]
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_negative_subset(
    examples: list[tuple[str, int]],
    record_id: str,
    negative_ratio: Optional[float],
) -> list[tuple[str, int]]:
    if negative_ratio is None:
        return examples
    positives = [example for example in examples if example[1] > 0]
    negatives = [example for example in examples if example[1] == 0]
    keep = min(len(negatives), max(1, math.ceil(len(positives) * negative_ratio)))
    negatives.sort(
        key=lambda example: hashlib.sha256(
            f"{record_id}\n{example[0]}".encode()
        ).digest()
    )
    return positives + negatives[:keep]


def _examples(
    records: Sequence[dict[str, Any]],
    split: str,
    negative_ratio: Optional[float] = None,
) -> tuple[list[str], torch.Tensor]:
    pair_texts: list[str] = []
    labels: list[int] = []
    for record in records:
        if _record_split(record["record_id"]) != split:
            continue
        nodes = record["nodes"]
        node_ids = [int(node["step_id"]) for node in nodes]
        text_by_id = {int(node["step_id"]): str(node["raw_text"]) for node in nodes}
        directed = {(int(edge["source"]), int(edge["target"])) for edge in record["edges"]}
        record_examples: list[tuple[str, int]] = []
        for order_index in _orders(len(nodes)):
            ordered_ids = [node_ids[index] for index in order_index]
            for left in range(len(ordered_ids)):
                for right in range(left + 1, len(ordered_ids)):
                    left_id, right_id = ordered_ids[left], ordered_ids[right]
                    has_forward = (left_id, right_id) in directed
                    has_backward = (right_id, left_id) in directed
                    if has_forward and has_backward:
                        continue
                    label = 1 if has_forward else 2 if has_backward else 0
                    record_examples.append(
                        (
                            format_pair_prompt(text_by_id[left_id], text_by_id[right_id]),
                            label,
                        )
                    )
        for text, label in _stable_negative_subset(
            record_examples,
            record["record_id"],
            negative_ratio,
        ):
            pair_texts.append(text)
            labels.append(label)
    if not pair_texts:
        raise RuntimeError(f"No {split} pair examples were constructed")
    return pair_texts, torch.tensor(labels, dtype=torch.long)


def _metrics(labels: torch.Tensor, predictions: torch.Tensor) -> dict[str, Any]:
    labels = labels.cpu()
    predictions = predictions.cpu()
    per_class: dict[str, dict[str, float | int]] = {}
    f1_values: list[float] = []
    for label, name in enumerate(LABEL_NAMES):
        true_positive = int(((labels == label) & (predictions == label)).sum())
        false_positive = int(((labels != label) & (predictions == label)).sum())
        false_negative = int(((labels == label) & (predictions != label)).sum())
        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
        recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        f1_values.append(f1)
        per_class[name] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": int((labels == label).sum()),
        }
    gold_edge = labels > 0
    predicted_edge = predictions > 0
    edge_true_positive = int((gold_edge & predicted_edge).sum())
    predicted_edge_count = int(predicted_edge.sum())
    gold_edge_count = int(gold_edge.sum())
    return {
        "accuracy": float((labels == predictions).float().mean()),
        "macro_f1": sum(f1_values) / len(f1_values),
        "edge_precision": edge_true_positive / predicted_edge_count if predicted_edge_count else 0.0,
        "edge_recall": edge_true_positive / gold_edge_count if gold_edge_count else 0.0,
        "direction_accuracy_on_gold_edges": (
            float((predictions[gold_edge] == labels[gold_edge]).float().mean())
            if bool(gold_edge.any())
            else 0.0
        ),
        "per_class": per_class,
    }


def _tokenize(
    tokenizer: Any,
    texts: Sequence[str],
    max_length: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    return tokenizer(
        list(texts),
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    ).to(device)


def _predict(
    model: torch.nn.Module,
    tokenizer: Any,
    texts: Sequence[str],
    device: torch.device,
    batch_size: int,
    max_length: int,
) -> torch.Tensor:
    chunks: list[torch.Tensor] = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            inputs = _tokenize(tokenizer, texts[start : start + batch_size], max_length, device)
            logits = model(**inputs).logits
            chunks.append(logits.float().cpu())
    return torch.cat(chunks)


def _threshold_predictions(probabilities: torch.Tensor, threshold: float, margin: float) -> torch.Tensor:
    predictions = probabilities.argmax(dim=-1)
    confidence = probabilities.gather(1, predictions[:, None]).squeeze(1)
    direction_margin = (probabilities[:, 1] - probabilities[:, 2]).abs()
    abstain = (predictions > 0) & ((confidence < threshold) | (direction_margin < margin))
    predictions[abstain] = 0
    return predictions


def _trainable_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }


def _restore_trainable_state(model: torch.nn.Module, state: dict[str, torch.Tensor]) -> None:
    parameters = dict(model.named_parameters())
    with torch.no_grad():
        for name, value in state.items():
            parameters[name].copy_(value.to(parameters[name].device))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path, nargs="+")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--field", default="semantic_teacher_dag")
    parser.add_argument("--backbone-model", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--train-batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--gradient-accumulation", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=384)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--negative-ratio", type=float, default=2.0)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0, help="Single canonical initialization seed.")
    parser.add_argument(
        "--reject-malformed-nodes",
        action="store_true",
        help="drop an entire teacher source if its step segmentation contains an obvious fragment",
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    candidate_records = _load_teacher_records(args.input, args.field)
    rejected_records = {}
    for record in candidate_records:
        reasons = _malformed_node_reasons(record)
        if reasons:
            rejected_records[record["record_id"]] = reasons
    records = (
        [record for record in candidate_records if record["record_id"] not in rejected_records]
        if args.reject_malformed_nodes
        else candidate_records
    )
    if len(records) < 10:
        raise RuntimeError(f"Only {len(records)} usable semantic-teacher records")
    train_texts, train_labels = _examples(records, "train", args.negative_ratio)
    validation_texts, validation_labels = _examples(records, "validation")

    transformers = _prepare_transformers()
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoTokenizer

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.backbone_model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    model_kwargs: dict[str, Any] = {"num_labels": 3, "trust_remote_code": True}
    major_version = int(str(transformers.__version__).split(".", 1)[0])
    model_kwargs["dtype" if major_version >= 5 else "torch_dtype"] = dtype
    base_model = _load_sequence_classifier(str(args.backbone_model), model_kwargs)
    base_model.config.pad_token_id = tokenizer.pad_token_id
    base_model.config.use_cache = False
    if hasattr(base_model.config, "text_config"):
        base_model.config.text_config.pad_token_id = tokenizer.pad_token_id
        base_model.config.text_config.use_cache = False
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "in_proj_qkv",
            "out_proj",
        ],
        modules_to_save=["score"],
    )
    model = get_peft_model(base_model, lora_config)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.to(device)
    model.print_trainable_parameters()

    counts = torch.bincount(train_labels, minlength=3).float()
    class_weights = len(train_labels) / (3.0 * counts.clamp_min(1.0))
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    generator = torch.Generator().manual_seed(args.seed)
    best_score = -math.inf
    best_epoch = 0
    best_state: dict[str, torch.Tensor] = {}
    stale = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        order = torch.randperm(len(train_labels), generator=generator)
        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0
        batches = math.ceil(len(order) / args.train_batch_size)
        for batch_number, start in enumerate(range(0, len(order), args.train_batch_size), start=1):
            indices = order[start : start + args.train_batch_size]
            texts = [train_texts[index] for index in indices.tolist()]
            inputs = _tokenize(tokenizer, texts, args.max_length, device)
            labels = train_labels[indices].to(device)
            logits = model(**inputs).logits
            loss = criterion(logits.float(), labels) / args.gradient_accumulation
            loss.backward()
            running_loss += float(loss.detach()) * args.gradient_accumulation
            should_step = batch_number % args.gradient_accumulation == 0 or batch_number == batches
            if should_step:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        validation_logits = _predict(
            model,
            tokenizer,
            validation_texts,
            device,
            args.eval_batch_size,
            args.max_length,
        )
        score = _metrics(validation_labels, validation_logits.argmax(dim=-1))["macro_f1"]
        print(
            json.dumps(
                {
                    "epoch": epoch,
                    "train_loss": running_loss / batches,
                    "validation_macro_f1": score,
                }
            ),
            flush=True,
        )
        if score > best_score + 1e-6:
            best_score = score
            best_epoch = epoch
            best_state = _trainable_state(model)
            stale = 0
        else:
            stale += 1
            if stale >= args.patience:
                break

    _restore_trainable_state(model, best_state)
    validation_logits = _predict(
        model,
        tokenizer,
        validation_texts,
        device,
        args.eval_batch_size,
        args.max_length,
    )
    probabilities = torch.softmax(validation_logits, dim=-1)
    best_calibration: tuple[float, float, float] = (-math.inf, 0.60, 0.15)
    for threshold in (0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70):
        for margin in (0.00, 0.05, 0.10, 0.15, 0.20):
            predictions = _threshold_predictions(probabilities.clone(), threshold, margin)
            score = _metrics(validation_labels, predictions)["macro_f1"]
            candidate = (score, threshold, margin)
            if candidate > best_calibration:
                best_calibration = candidate
    _, edge_threshold, direction_margin = best_calibration
    final_predictions = _threshold_predictions(
        probabilities.clone(),
        edge_threshold,
        direction_margin,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output_dir, safe_serialization=True)
    tokenizer.save_pretrained(args.output_dir)
    config = {
        "schema_version": "topoprm.edge_encoder.v2",
        "backend": "qwen_sequence_classifier_lora",
        "backbone_model": str(args.backbone_model),
        "max_length": args.max_length,
        "labels": list(LABEL_NAMES),
        "edge_threshold": edge_threshold,
        "direction_margin": direction_margin,
        "pair_prompt_version": PAIR_PROMPT_VERSION,
        "lora_rank": args.lora_rank,
        "lora_alpha": args.lora_alpha,
        "teacher_inputs": [
            {"path": str(path.resolve()), "sha256": _sha256(path)}
            for path in args.input
        ],
        "training_node_filter": (
            "shared_strict_segmentation_quality_gate"
            if args.reject_malformed_nodes
            else "none"
        ),
    }
    (args.output_dir / "edge_encoder_config.json").write_text(
        json.dumps(config, indent=2) + "\n",
        encoding="utf-8",
    )
    metrics = {
        "canonical_seed": args.seed,
        "best_epoch": best_epoch,
        "candidate_source_records": len(candidate_records),
        "rejected_malformed_source_records": (
            len(rejected_records) if args.reject_malformed_nodes else 0
        ),
        "rejected_reason_counts": (
            dict(Counter(reason for reasons in rejected_records.values() for reason in reasons))
            if args.reject_malformed_nodes
            else {}
        ),
        "usable_source_records": len(records),
        "train_source_records": sum(_record_split(record["record_id"]) == "train" for record in records),
        "validation_source_records": sum(
            _record_split(record["record_id"]) == "validation" for record in records
        ),
        "train_pairs": len(train_labels),
        "validation_pairs": len(validation_labels),
        "train_class_counts": dict(Counter(int(label) for label in train_labels.tolist())),
        "validation_class_counts": dict(Counter(int(label) for label in validation_labels.tolist())),
        "edge_threshold": edge_threshold,
        "direction_margin": direction_margin,
        "validation": _metrics(validation_labels, final_predictions),
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metrics, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

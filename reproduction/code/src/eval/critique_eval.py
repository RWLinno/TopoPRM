"""Evaluator for critique quality on math-grading tasks."""
from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional


@dataclass
class CritiqueMetrics:
    score_accuracy: float = 0.0
    error_identification_precision: float = 0.0
    error_identification_recall: float = 0.0
    error_identification_f1: float = 0.0
    step_coverage: float = 0.0
    format_compliance: float = 0.0
    avg_prediction_tokens: float = 0.0
    num_samples: int = 0


class CritiqueEvaluator:
    """Evaluate quality of model-generated math critique outputs.

    This version supports multiple field schemas observed in this project:
    - prediction text keys: prediction / output / response
    - ground-truth score keys: score / 学生得分 / std_score / solution
    - optional step fields: error_steps / step_results / steps
    """

    def __init__(self) -> None:
        self._predictions: list[dict[str, Any]] = []
        self._ground_truths: list[dict[str, Any]] = []

    @staticmethod
    def _load_jsonl(path: str | Path) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records

    def load(self, predictions_path: str | Path, ground_truth_path: str | Path) -> None:
        self._predictions = self._load_jsonl(predictions_path)
        self._ground_truths = self._load_jsonl(ground_truth_path)

    @staticmethod
    def _extract_text(record: dict[str, Any]) -> str:
        return str(record.get("prediction") or record.get("output") or record.get("response") or "")

    @staticmethod
    def _approx_token_count(text: str) -> int:
        # A lightweight, tokenizer-free approximation:
        # - English words/digits count as one token
        # - each CJK character counts as one token
        # - remaining non-space symbols count as one token
        parts = re.findall(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]|[^\s]", text)
        return len(parts)

    @staticmethod
    def _extract_answer_json(text: str) -> Optional[dict[str, Any]]:
        m = re.search(r"<answer>\s*(.*?)\s*</answer>", text, re.DOTALL)
        if not m:
            return None
        payload = m.group(1)
        try:
            return json.loads(payload)
        except json.JSONDecodeError:
            # last-resort cleanup: trim trailing commas before braces/brackets
            cleaned = re.sub(r",\s*([}\]])", r"\1", payload)
            try:
                return json.loads(cleaned)
            except json.JSONDecodeError:
                return None

    @staticmethod
    def _to_float(v: Any) -> Optional[float]:
        try:
            if v is None:
                return None
            return float(v)
        except (TypeError, ValueError):
            return None

    def _extract_score(self, record: dict[str, Any]) -> Optional[float]:
        # direct keys
        for key in ("score", "学生得分", "得分", "std_score", "solution"):
            if key in record:
                sv = self._to_float(record.get(key))
                if sv is not None:
                    return sv

        # parse from answer json in text fields
        text = self._extract_text(record)
        ans = self._extract_answer_json(text)
        if ans is not None:
            for key in ("score", "学生得分", "得分"):
                if key in ans:
                    sv = self._to_float(ans.get(key))
                    if sv is not None:
                        return sv
        return None

    @staticmethod
    def _extract_error_steps(record: dict[str, Any]) -> set[int]:
        # Explicit schema
        raw = record.get("error_steps")
        if isinstance(raw, list):
            out = set()
            for x in raw:
                try:
                    out.add(int(x))
                except Exception:
                    pass
            return out

        # Ground-truth schema: step_results may contain correctness labels
        step_results = record.get("step_results")
        if isinstance(step_results, list):
            errs = set()
            for i, item in enumerate(step_results, start=1):
                txt = str(item)
                if any(t in txt for t in ["错误", "错", "incorrect", "False"]):
                    errs.add(i)
            return errs

        return set()

    @staticmethod
    def _extract_steps(record: dict[str, Any]) -> list[str]:
        if isinstance(record.get("steps"), list):
            return [str(x) for x in record["steps"]]
        if isinstance(record.get("user_step_split_emb"), list):
            return [str(x) for x in record["user_step_split_emb"]]
        return []


    @staticmethod
    def _pair_records(preds: list[dict[str, Any]], gts: list[dict[str, Any]]) -> list[tuple[dict[str, Any], dict[str, Any]]]:
        """Pair prediction/ground-truth records by id when available, otherwise by index."""
        gt_map = {g.get("id"): g for g in gts if g.get("id") is not None}
        pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []

        # try id-based pairing first
        id_hits = 0
        for p in preds:
            pid = p.get("id")
            if pid is not None and pid in gt_map:
                pairs.append((p, gt_map[pid]))
                id_hits += 1

        # if almost no ids exist in predictions, fallback to positional pairing
        if id_hits == 0:
            m = min(len(preds), len(gts))
            pairs = [(preds[i], gts[i]) for i in range(m)]

        return pairs

    def score_accuracy(self, preds: list[dict[str, Any]], gts: list[dict[str, Any]]) -> float:
        pairs = self._pair_records(preds, gts)
        correct = total = 0
        for p, gt in pairs:
            total += 1
            ps = self._extract_score(p)
            gs = self._extract_score(gt)
            if ps is not None and gs is not None and ps == gs:
                correct += 1
        return correct / total if total else 0.0

    def error_identification_prf(self, preds: list[dict[str, Any]], gts: list[dict[str, Any]]) -> tuple[float, float, float]:
        pairs = self._pair_records(preds, gts)
        tp = fp = fn = 0
        for p, gt in pairs:
            pred_set = self._extract_error_steps(p)
            gt_set = self._extract_error_steps(gt)
            tp += len(pred_set & gt_set)
            fp += len(pred_set - gt_set)
            fn += len(gt_set - pred_set)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        return precision, recall, f1

    def average_prediction_tokens(self, preds: list[dict[str, Any]]) -> float:
        if not preds:
            return 0.0
        total = 0
        for p in preds:
            total += self._approx_token_count(self._extract_text(p))
        return total / len(preds)

    def step_coverage(self, preds: list[dict[str, Any]], gts: list[dict[str, Any]]) -> float:
        pairs = self._pair_records(preds, gts)
        ratios: list[float] = []
        for p, gt in pairs:
            gt_steps = self._extract_steps(gt)
            if not gt_steps:
                continue
            pred_text = self._extract_text(p)
            covered = sum(1 for s in gt_steps if s.strip() and s.strip() in pred_text)
            ratios.append(covered / len(gt_steps))
        return sum(ratios) / len(ratios) if ratios else 0.0

    def format_compliance(self, preds: list[dict[str, Any]]) -> float:
        compliant = 0
        for p in preds:
            text = self._extract_text(p)
            has_think = bool(re.search(r"<think>.*?</think>", text, re.DOTALL))
            ans = self._extract_answer_json(text)
            if has_think and ans is not None:
                compliant += 1
        return compliant / len(preds) if preds else 0.0

    def evaluate(
        self,
        predictions: Optional[list[dict[str, Any]]] = None,
        ground_truths: Optional[list[dict[str, Any]]] = None,
    ) -> CritiqueMetrics:
        preds = predictions if predictions is not None else self._predictions
        gts = ground_truths if ground_truths is not None else self._ground_truths
        precision, recall, f1 = self.error_identification_prf(preds, gts)
        return CritiqueMetrics(
            score_accuracy=self.score_accuracy(preds, gts),
            error_identification_precision=precision,
            error_identification_recall=recall,
            error_identification_f1=f1,
            step_coverage=self.step_coverage(preds, gts),
            format_compliance=self.format_compliance(preds),
            avg_prediction_tokens=self.average_prediction_tokens(preds),
            num_samples=len(preds),
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate critique quality against ground truth")
    parser.add_argument("--predictions", required=True, help="Path to predictions JSONL file")
    parser.add_argument("--ground_truth", required=True, help="Path to ground-truth JSONL file")
    parser.add_argument("--output", default=None, help="Optional path to write metrics JSON")
    args = parser.parse_args()

    evaluator = CritiqueEvaluator()
    evaluator.load(args.predictions, args.ground_truth)
    metrics = evaluator.evaluate()

    result = {
        "score_accuracy": metrics.score_accuracy,
        "error_identification_precision": metrics.error_identification_precision,
        "error_identification_recall": metrics.error_identification_recall,
        "error_identification_f1": metrics.error_identification_f1,
        "step_coverage": metrics.step_coverage,
        "format_compliance": metrics.format_compliance,
        "avg_prediction_tokens": metrics.avg_prediction_tokens,
        "num_samples": metrics.num_samples,
    }

    print(json.dumps(result, indent=2, ensure_ascii=False))

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"\nMetrics written to {args.output}")


if __name__ == "__main__":
    main()

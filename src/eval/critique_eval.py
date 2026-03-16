"""Evaluator for critique quality on math-grading tasks."""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


@dataclass
class CritiqueMetrics:
    """Aggregated critique-quality metrics."""

    score_accuracy: float = 0.0
    error_identification_f1: float = 0.0
    step_coverage: float = 0.0
    format_compliance: float = 0.0
    num_samples: int = 0


class CritiqueEvaluator:
    """Evaluate the quality of model-generated critiques.

    The evaluator loads prediction and ground-truth JSONL files and
    computes four metrics:

    * **score_accuracy** – fraction of samples where the predicted student
      score exactly matches the ground truth.
    * **error_identification_f1** – micro-averaged F1 over the sets of
      identified error step indices.
    * **step_coverage** – average ratio of ground-truth steps that appear
      (textually) in the predicted critique.
    * **format_compliance** – fraction of predictions that contain both
      ``<think>`` and ``<answer>`` blocks with parseable JSON.

    Both files are in **JSONL** format.  Each line must be a JSON object
    with at least an ``"id"`` field.

    Expected prediction fields
    --------------------------
    * ``id`` – sample identifier
    * ``prediction`` or ``output`` – the raw model output string
    * ``score`` or ``学生得分`` – predicted score (optional, can be inside
      ``<answer>`` JSON)

    Expected ground-truth fields
    ----------------------------
    * ``id`` – sample identifier
    * ``score`` or ``学生得分`` – ground-truth score
    * ``error_steps`` – list of step indices where errors occur
    * ``steps`` – list of step-text strings (for coverage computation)
    """

    def __init__(self) -> None:
        self._predictions: list[dict[str, Any]] = []
        self._ground_truths: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # I/O
    # ------------------------------------------------------------------

    @staticmethod
    def _load_jsonl(path: str | Path) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records

    def load(
        self,
        predictions_path: str | Path,
        ground_truth_path: str | Path,
    ) -> None:
        """Load prediction and ground-truth JSONL files."""
        self._predictions = self._load_jsonl(predictions_path)
        self._ground_truths = self._load_jsonl(ground_truth_path)

    # ------------------------------------------------------------------
    # individual metrics
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_score(record: dict[str, Any]) -> Optional[float]:
        """Try to get a numeric score from a record."""
        for key in ("score", "学生得分", "得分"):
            if key in record:
                try:
                    return float(record[key])
                except (TypeError, ValueError):
                    continue
        import re
        text = record.get("prediction") or record.get("output") or ""
        m = re.search(r"<answer>\s*(.*?)\s*</answer>", str(text), re.DOTALL)
        if m:
            try:
                obj = json.loads(m.group(1))
                for key in ("score", "学生得分", "得分"):
                    if key in obj:
                        return float(obj[key])
            except (json.JSONDecodeError, TypeError, ValueError):
                pass
        return None

    @staticmethod
    def _extract_error_steps(record: dict[str, Any]) -> set[int]:
        raw = record.get("error_steps", [])
        if isinstance(raw, list):
            return {int(x) for x in raw}
        return set()

    def score_accuracy(
        self,
        preds: list[dict[str, Any]],
        gts: list[dict[str, Any]],
    ) -> float:
        """Fraction of samples with an exact score match."""
        gt_map = {g["id"]: g for g in gts}
        correct = 0
        total = 0
        for p in preds:
            gt = gt_map.get(p.get("id"))
            if gt is None:
                continue
            total += 1
            pred_score = self._extract_score(p)
            gt_score = self._extract_score(gt)
            if pred_score is not None and gt_score is not None and pred_score == gt_score:
                correct += 1
        return correct / total if total else 0.0

    def error_identification_f1(
        self,
        preds: list[dict[str, Any]],
        gts: list[dict[str, Any]],
    ) -> float:
        """Micro-averaged F1 over error-step identification."""
        gt_map = {g["id"]: g for g in gts}
        tp_total = 0
        fp_total = 0
        fn_total = 0
        for p in preds:
            gt = gt_map.get(p.get("id"))
            if gt is None:
                continue
            pred_set = self._extract_error_steps(p)
            gt_set = self._extract_error_steps(gt)
            tp_total += len(pred_set & gt_set)
            fp_total += len(pred_set - gt_set)
            fn_total += len(gt_set - pred_set)
        precision = tp_total / (tp_total + fp_total) if (tp_total + fp_total) else 0.0
        recall = tp_total / (tp_total + fn_total) if (tp_total + fn_total) else 0.0
        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)

    def step_coverage(
        self,
        preds: list[dict[str, Any]],
        gts: list[dict[str, Any]],
    ) -> float:
        """Average ratio of ground-truth steps mentioned in the prediction."""
        gt_map = {g["id"]: g for g in gts}
        ratios: list[float] = []
        for p in preds:
            gt = gt_map.get(p.get("id"))
            if gt is None:
                continue
            gt_steps: list[str] = gt.get("steps", [])
            if not gt_steps:
                continue
            pred_text = str(p.get("prediction") or p.get("output") or "")
            covered = sum(1 for s in gt_steps if s.strip() and s.strip() in pred_text)
            ratios.append(covered / len(gt_steps))
        return sum(ratios) / len(ratios) if ratios else 0.0

    def format_compliance(self, preds: list[dict[str, Any]]) -> float:
        """Fraction of predictions with ``<think>`` + ``<answer>`` + valid JSON."""
        import re

        compliant = 0
        for p in preds:
            text = str(p.get("prediction") or p.get("output") or "")
            has_think = bool(re.search(r"<think>.*?</think>", text, re.DOTALL))
            m = re.search(r"<answer>\s*(.*?)\s*</answer>", text, re.DOTALL)
            has_answer_json = False
            if m:
                try:
                    json.loads(m.group(1))
                    has_answer_json = True
                except json.JSONDecodeError:
                    pass
            if has_think and has_answer_json:
                compliant += 1
        return compliant / len(preds) if preds else 0.0

    # ------------------------------------------------------------------
    # aggregate
    # ------------------------------------------------------------------

    def evaluate(
        self,
        predictions: Optional[list[dict[str, Any]]] = None,
        ground_truths: Optional[list[dict[str, Any]]] = None,
    ) -> CritiqueMetrics:
        """Compute all metrics and return a :class:`CritiqueMetrics` object."""
        preds = predictions if predictions is not None else self._predictions
        gts = ground_truths if ground_truths is not None else self._ground_truths

        return CritiqueMetrics(
            score_accuracy=self.score_accuracy(preds, gts),
            error_identification_f1=self.error_identification_f1(preds, gts),
            step_coverage=self.step_coverage(preds, gts),
            format_compliance=self.format_compliance(preds),
            num_samples=len(preds),
        )


def main() -> None:
    """CLI entry-point for critique evaluation."""
    parser = argparse.ArgumentParser(
        description="Evaluate critique quality against ground truth",
    )
    parser.add_argument(
        "--predictions", required=True,
        help="Path to predictions JSONL file",
    )
    parser.add_argument(
        "--ground_truth", required=True,
        help="Path to ground-truth JSONL file",
    )
    parser.add_argument(
        "--output", default=None,
        help="Optional path to write metrics JSON",
    )
    args = parser.parse_args()

    evaluator = CritiqueEvaluator()
    evaluator.load(args.predictions, args.ground_truth)
    metrics = evaluator.evaluate()

    result = {
        "score_accuracy": metrics.score_accuracy,
        "error_identification_f1": metrics.error_identification_f1,
        "step_coverage": metrics.step_coverage,
        "format_compliance": metrics.format_compliance,
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

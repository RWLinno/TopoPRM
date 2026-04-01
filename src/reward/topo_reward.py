from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any, Optional

from swift.rewards import ORM, orms

from src.dag.graph import ReasoningDAG
from src.data.build_dag import build_dag_from_answer, extract_steps_from_answer


@dataclass
class TopoVerification:
    """Deterministic, auditable components for topological process reward."""

    valid_dag: float
    acyclic: float
    no_orphan: float
    direction_consistency: float
    step_alignment: float
    ref_edge_precision: float
    ref_edge_recall: float
    ref_edge_f1: float

    def as_dict(self) -> dict[str, float]:
        return {
            'valid_dag': self.valid_dag,
            'acyclic': self.acyclic,
            'no_orphan': self.no_orphan,
            'direction_consistency': self.direction_consistency,
            'step_alignment': self.step_alignment,
            'ref_edge_precision': self.ref_edge_precision,
            'ref_edge_recall': self.ref_edge_recall,
            'ref_edge_f1': self.ref_edge_f1,
        }


class TopoReward(ORM):
    """Verifiable topology-aware reward for reasoning traces.

    This reward decomposes R_topo into auditable deterministic components and
    computes a normalized weighted score. The design goal is to make every
    reward point traceable to graph properties rather than opaque heuristics.
    """

    W_VALID: float = float(os.environ.get('TOPO_W_VALID', '0.20') or 0.20)
    W_ACYCLIC: float = float(os.environ.get('TOPO_W_ACYCLIC', '0.15') or 0.15)
    W_NO_ORPHAN: float = float(os.environ.get('TOPO_W_NO_ORPHAN', '0.15') or 0.15)
    W_DIRECTION: float = float(os.environ.get('TOPO_W_DIRECTION', '0.15') or 0.15)
    W_STEP_ALIGN: float = float(os.environ.get('TOPO_W_STEP_ALIGN', '0.10') or 0.10)
    W_REF_EDGE_F1: float = float(os.environ.get('TOPO_W_REF_EDGE_F1', '0.25') or 0.25)

    REQUIRE_VALID_DAG: bool = (os.environ.get('TOPO_REQUIRE_VALID_DAG', '1') or '1') != '0'
    LOG_EVERY: int = int(os.environ.get('TOPO_VERIFY_LOG_EVERY', '0') or 0)

    def __init__(self, **kwargs: Any) -> None:
        super().__init__()
        self._num_calls = 0

    @staticmethod
    def _extract_think(text: str) -> str:
        m = re.search(r'<think>(.*?)</think>', text, re.DOTALL)
        return m.group(1).strip() if m else ''

    @staticmethod
    def _safe_clip01(v: float) -> float:
        return max(0.0, min(1.0, float(v)))

    @staticmethod
    def _orphan_conclusion_ratio(dag: ReasoningDAG) -> float:
        from src.dag.node import StepType

        conclusion_ids = [
            sid for sid, n in dag.nodes.items() if n.step_type == StepType.CONCLUSION
        ]
        if not conclusion_ids:
            return 0.0
        orphan_count = 0
        for cid in conclusion_ids:
            has_virtual_pred = any(
                dag.is_virtual_edge(dag.graph.edges[u, cid].get('edge_type', ''))
                for u in dag.graph.predecessors(cid)
            )
            if not has_virtual_pred:
                orphan_count += 1
        return orphan_count / len(conclusion_ids)

    @staticmethod
    def _step_alignment(dag: ReasoningDAG, num_steps: int) -> float:
        if num_steps <= 0:
            return 0.0
        # 1.0 when DAG node count matches extracted step count.
        delta = abs(dag.num_nodes - num_steps) / max(num_steps, 1)
        return max(0.0, 1.0 - delta)

    @staticmethod
    def _virtual_edges(dag: ReasoningDAG) -> set[tuple[str, str]]:
        return {
            (u, v)
            for u, v, d in dag.graph.edges(data=True)
            if dag.is_virtual_edge(d.get('edge_type', ''))
        }

    def _ref_edge_prf(self, dag: ReasoningDAG, ref_dag: Optional[ReasoningDAG]) -> tuple[float, float, float]:
        if ref_dag is None:
            return 0.0, 0.0, 0.0

        pred = self._virtual_edges(dag)
        gold = self._virtual_edges(ref_dag)
        if not gold and not pred:
            return 1.0, 1.0, 1.0
        if not pred:
            return 0.0, 0.0, 0.0

        tp = len(pred & gold)
        precision = tp / len(pred) if pred else 0.0
        recall = tp / len(gold) if gold else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        return precision, recall, f1

    def _compute_verification(
        self,
        dag: ReasoningDAG,
        num_steps: int,
        ref_dag: Optional[ReasoningDAG],
    ) -> TopoVerification:
        valid = 1.0 if dag.is_valid_dag() else 0.0
        acyclic = 1.0 if dag.validate_dag().get('is_acyclic', False) else 0.0
        no_orphan = 1.0 - self._orphan_conclusion_ratio(dag)
        direction = dag.direction_consistency()
        step_align = self._step_alignment(dag, num_steps)
        p, r, f1 = self._ref_edge_prf(dag, ref_dag)

        return TopoVerification(
            valid_dag=self._safe_clip01(valid),
            acyclic=self._safe_clip01(acyclic),
            no_orphan=self._safe_clip01(no_orphan),
            direction_consistency=self._safe_clip01(direction),
            step_alignment=self._safe_clip01(step_align),
            ref_edge_precision=self._safe_clip01(p),
            ref_edge_recall=self._safe_clip01(r),
            ref_edge_f1=self._safe_clip01(f1),
        )

    def _score(self, v: TopoVerification, has_ref: bool) -> float:
        weights = {
            'valid': self.W_VALID,
            'acyclic': self.W_ACYCLIC,
            'no_orphan': self.W_NO_ORPHAN,
            'direction': self.W_DIRECTION,
            'step_align': self.W_STEP_ALIGN,
            'ref_f1': self.W_REF_EDGE_F1 if has_ref else 0.0,
        }
        denom = sum(max(0.0, w) for w in weights.values())
        if denom <= 1e-12:
            return 0.0

        total = (
            weights['valid'] * v.valid_dag
            + weights['acyclic'] * v.acyclic
            + weights['no_orphan'] * v.no_orphan
            + weights['direction'] * v.direction_consistency
            + weights['step_align'] * v.step_alignment
            + weights['ref_f1'] * v.ref_edge_f1
        )
        score = total / denom
        if self.REQUIRE_VALID_DAG and v.valid_dag < 1.0:
            # Hard gate: invalid DAG gets zero topology credit.
            return 0.0
        return self._safe_clip01(score)

    def _parse_ref(self, raw: Any) -> Optional[ReasoningDAG]:
        if raw is None:
            return None
        try:
            if isinstance(raw, ReasoningDAG):
                return raw
            if isinstance(raw, dict):
                return ReasoningDAG.from_dict(raw)
            if isinstance(raw, str) and raw.strip():
                return ReasoningDAG.from_json(raw)
        except Exception:
            return None
        return None

    def __call__(
        self,
        completions: list,
        reference_dag: Optional[Any] = None,
        **kwargs: Any,
    ) -> list[float]:
        refs = reference_dag if isinstance(reference_dag, list) else [reference_dag] * len(completions)

        rewards: list[float] = []
        diag_rows: list[dict[str, float]] = []
        for idx, completion in enumerate(completions):
            text = completion if isinstance(completion, str) else (completion[-1].get('content', '') if completion else '')
            think_text = self._extract_think(text)
            steps = extract_steps_from_answer(think_text)
            if not steps:
                rewards.append(0.0)
                continue

            dag = build_dag_from_answer(think_text)
            ref = self._parse_ref(refs[idx] if idx < len(refs) else None)
            verification = self._compute_verification(dag, len(steps), ref)
            reward = self._score(verification, has_ref=(ref is not None))
            rewards.append(reward)

            row = verification.as_dict()
            row['r_topo'] = reward
            diag_rows.append(row)

        if self.LOG_EVERY:
            self._num_calls += 1
            if self._num_calls % self.LOG_EVERY == 0 and diag_rows:
                mean = lambda k: sum(x.get(k, 0.0) for x in diag_rows) / len(diag_rows)
                print(
                    '[topo_verify] '
                    f"valid={mean('valid_dag'):.3f} "
                    f"acyclic={mean('acyclic'):.3f} "
                    f"no_orphan={mean('no_orphan'):.3f} "
                    f"dir={mean('direction_consistency'):.3f} "
                    f"step_align={mean('step_alignment'):.3f} "
                    f"ref_f1={mean('ref_edge_f1'):.3f} "
                    f"r_topo={mean('r_topo'):.3f} n={len(diag_rows)}"
                )

        return rewards


orms['topo_reward'] = TopoReward

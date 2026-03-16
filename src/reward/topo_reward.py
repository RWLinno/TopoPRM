from __future__ import annotations

import re
from typing import Any, Optional

from swift.rewards import ORM, orms

from src.dag.graph import ReasoningDAG
from src.data.build_dag import build_dag_from_answer, extract_steps_from_answer


class TopoReward(ORM):
    """Topological-structure validity reward (weak supervision, **core innovation**).

    Extracts reasoning steps from the ``<think>`` block, builds a
    :class:`ReasoningDAG`, and evaluates its structural quality.

    Scoring rubric (additive, max ≈ 1.0)
    -------------------------------------
    * Extractable steps present → **base 0.4**
    * DAG is acyclic → **+0.2**
    * No orphan conclusions (every conclusion node has ≥1 dependency
      predecessor) → **+0.2**
    * Consistent direction (no conclusion→non-conclusion back-edges) →
      **+0.1**
    * Reference-DAG key-dependency coverage × **0.1** (optional; enabled
      when *reference_dag* kwarg is supplied)
    """

    @staticmethod
    def _extract_think(text: str) -> str:
        """Return content inside ``<think>…</think>``, or empty string."""
        m = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
        return m.group(1).strip() if m else ""

    @staticmethod
    def _orphan_conclusion_ratio(dag: ReasoningDAG) -> float:
        """Fraction of conclusion nodes that lack dependency predecessors."""
        from src.dag.node import StepType

        conclusion_ids = [
            sid for sid, n in dag.nodes.items()
            if n.step_type == StepType.CONCLUSION
        ]
        if not conclusion_ids:
            return 0.0
        orphan_count = 0
        for cid in conclusion_ids:
            has_dep_pred = any(
                dag.graph.edges[u, cid].get("edge_type") == "dependency"
                for u in dag.graph.predecessors(cid)
            )
            if not has_dep_pred:
                orphan_count += 1
        return orphan_count / len(conclusion_ids)

    @staticmethod
    def _direction_consistency(dag: ReasoningDAG) -> float:
        """1.0 when no conclusion→non-conclusion back-edges exist."""
        return dag.direction_consistency()

    @staticmethod
    def _key_dep_coverage(dag: ReasoningDAG, ref_dag: ReasoningDAG) -> float:
        """Fraction of reference dependency edges covered in *dag*."""
        ref_deps = {
            (u, v)
            for u, v, d in ref_dag.graph.edges(data=True)
            if d.get("edge_type") == "dependency"
        }
        if not ref_deps:
            return 1.0
        dag_deps = {
            (u, v)
            for u, v, d in dag.graph.edges(data=True)
            if d.get("edge_type") == "dependency"
        }
        covered = ref_deps & dag_deps
        return len(covered) / len(ref_deps)

    def __call__(
        self,
        completions: list,
        reference_dag: Optional[Any] = None,
        **kwargs: Any,
    ) -> list[float]:
        """Return topological-structure reward in [0, 1] per completion."""
        ref_dags = reference_dag if isinstance(reference_dag, list) else [reference_dag] * len(completions)

        rewards: list[float] = []
        for idx, completion in enumerate(completions):
            ref: Optional[ReasoningDAG] = None
            ref_raw = ref_dags[idx] if idx < len(ref_dags) else None
            if ref_raw is not None:
                try:
                    if isinstance(ref_raw, ReasoningDAG):
                        ref = ref_raw
                    elif isinstance(ref_raw, dict):
                        ref = ReasoningDAG.from_dict(ref_raw)
                    elif isinstance(ref_raw, str) and ref_raw.strip():
                        ref = ReasoningDAG.from_json(ref_raw)
                except Exception:
                    ref = None
            text = completion if isinstance(completion, str) else (completion[-1].get("content", "") if completion else "")
            think_text = self._extract_think(text)

            steps = extract_steps_from_answer(think_text)
            if not steps:
                rewards.append(0.0)
                continue

            dag = build_dag_from_answer(think_text)
            score = 0.4  # base for extractable steps

            if dag.is_valid_dag():
                score += 0.2

            orphan_ratio = self._orphan_conclusion_ratio(dag)
            if orphan_ratio == 0.0:
                score += 0.2

            dir_cons = self._direction_consistency(dag)
            score += 0.1 * dir_cons

            if ref is not None:
                coverage = self._key_dep_coverage(dag, ref)
                score += 0.1 * coverage

            rewards.append(min(score, 1.0))

        return rewards

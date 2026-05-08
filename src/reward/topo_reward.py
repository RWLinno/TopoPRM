from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from swift.rewards import ORM, orms

import os

from src.dag.graph import ReasoningDAG
from src.data.build_dag import build_dag_from_answer, extract_steps_from_answer
from src.reward.reward_config import RewardConfig
from src.reward.utils import completion_to_text, extract_think_block

# Edge weights used by the weighted orphan-support score.
# - virtual edges (claim_ref / expr_ref / var_ref) are strong supports → 1.0
# - double_barrier edges (auto fallback "weak support") → 0.5
# - solid edges (sequential ordering) → 0.3
# A conclusion is "fully supported" once the weight sum of its predecessors ≥ 1.0.
_ORPHAN_W_VIRTUAL: float = float(os.environ.get("TOPO_ORPHAN_W_VIRTUAL", 1.0))
_ORPHAN_W_DOUBLE_BARRIER: float = float(os.environ.get("TOPO_ORPHAN_W_DOUBLE_BARRIER", 0.5))
_ORPHAN_W_SOLID: float = float(os.environ.get("TOPO_ORPHAN_W_SOLID", 0.3))


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


@dataclass
class TopoFormulaTerms:
    lambda_base: float
    lambda_acyclic: float
    lambda_orphan: float
    lambda_delta: float
    lambda_kappa: float
    indicator_non_empty: float
    indicator_acyclic: float
    indicator_no_orphan: float
    rho_orphan: float
    delta: float
    kappa: float
    term_base: float
    term_acyclic: float
    term_orphan: float
    term_delta: float
    term_kappa: float
    denom: float
    r_topo: float

    def as_dict(self) -> dict[str, float]:
        return {
            "lambda_base": self.lambda_base,
            "lambda_acyclic": self.lambda_acyclic,
            "lambda_orphan": self.lambda_orphan,
            "lambda_delta": self.lambda_delta,
            "lambda_kappa": self.lambda_kappa,
            "indicator_non_empty": self.indicator_non_empty,
            "indicator_acyclic": self.indicator_acyclic,
            "indicator_no_orphan": self.indicator_no_orphan,
            "rho_orphan": self.rho_orphan,
            "delta": self.delta,
            "kappa": self.kappa,
            "term_base": self.term_base,
            "term_acyclic": self.term_acyclic,
            "term_orphan": self.term_orphan,
            "term_delta": self.term_delta,
            "term_kappa": self.term_kappa,
            "denom": self.denom,
            "r_topo": self.r_topo,
        }


class TopoReward(ORM):
    """Verifiable topology-aware reward for reasoning traces.

    This reward decomposes R_topo into auditable deterministic components and
    computes a normalized weighted score. The design goal is to make every
    reward point traceable to graph properties rather than opaque heuristics.
    """

    W_VALID: float = RewardConfig.TOPO_W_VALID
    W_ACYCLIC: float = RewardConfig.TOPO_W_ACYCLIC
    W_NO_ORPHAN: float = RewardConfig.TOPO_W_NO_ORPHAN
    W_DIRECTION: float = RewardConfig.TOPO_W_DIRECTION
    W_STEP_ALIGN: float = RewardConfig.TOPO_W_STEP_ALIGN
    W_REF_EDGE_F1: float = RewardConfig.TOPO_W_REF_EDGE_F1

    # Formula-aligned lambdas (Eq. r_topo)
    LAMBDA_BASE: float = RewardConfig.TOPO_LAMBDA_BASE
    LAMBDA_ACYCLIC: float = RewardConfig.TOPO_LAMBDA_ACYCLIC
    LAMBDA_ORPHAN: float = RewardConfig.TOPO_LAMBDA_ORPHAN
    LAMBDA_DELTA: float = RewardConfig.TOPO_LAMBDA_DELTA
    LAMBDA_KAPPA: float = RewardConfig.TOPO_LAMBDA_KAPPA

    REQUIRE_VALID_DAG: bool = RewardConfig.TOPO_REQUIRE_VALID_DAG
    LOG_EVERY: int = RewardConfig.TOPO_VERIFY_LOG_EVERY

    def __init__(self, **kwargs: Any) -> None:
        super().__init__()
        self._num_calls = 0
        self.last_diagnostics: list[dict[str, float]] = []

    @staticmethod
    def _safe_clip01(v: float) -> float:
        return max(0.0, min(1.0, float(v)))

    @staticmethod
    def _orphan_conclusion_ratio_legacy(dag: ReasoningDAG) -> float:
        """Legacy binary orphan ratio (pre-2026-04-23).

        A conclusion is "orphan" iff it has no virtual predecessor.
        Treats double_barrier and solid predecessors as zero support, which
        in v3b inflated rho_orphan and pinned topo reward to 0 on many
        rollouts.  Kept for ablation reproducibility; activate with env
        ``TOPO_ORPHAN_LEGACY=1``.
        """
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
    def _orphan_conclusion_ratio(dag: ReasoningDAG) -> float:
        """Weighted orphan ratio: rho ∈ [0, 1], lower is better.

        For each conclusion node we accumulate a support weight from its
        predecessors:
          virtual_edge → ``_ORPHAN_W_VIRTUAL`` (default 1.0)
          double_barrier_edge → ``_ORPHAN_W_DOUBLE_BARRIER`` (default 0.5)
          solid_edge → ``_ORPHAN_W_SOLID`` (default 0.3)

        A node is "fully supported" once weight ≥ 1.0; otherwise it
        contributes ``1 - support`` to the orphan ratio.  This converts
        the previously binary signal into a continuous one and stops
        treating fallback double_barrier edges as "no support".

        Set ``TOPO_ORPHAN_LEGACY=1`` to fall back to the binary version.
        """
        if os.environ.get("TOPO_ORPHAN_LEGACY", "0") not in ("0", "false", "False", ""):
            return TopoReward._orphan_conclusion_ratio_legacy(dag)

        from src.dag.node import StepType

        conclusion_ids = [
            sid for sid, n in dag.nodes.items() if n.step_type == StepType.CONCLUSION
        ]
        if not conclusion_ids:
            return 0.0

        orphan_score_sum = 0.0
        for cid in conclusion_ids:
            support = 0.0
            for u in dag.graph.predecessors(cid):
                etype = dag.graph.edges[u, cid].get('edge_type', '')
                if dag.is_virtual_edge(etype):
                    support += _ORPHAN_W_VIRTUAL
                elif dag.is_barrier_edge(etype):
                    support += _ORPHAN_W_DOUBLE_BARRIER
                elif dag.is_solid_edge(etype):
                    support += _ORPHAN_W_SOLID
            support = min(1.0, max(0.0, support))
            orphan_score_sum += (1.0 - support)
        return orphan_score_sum / len(conclusion_ids)

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

    def _compute_formula_terms(
        self,
        dag: ReasoningDAG,
        v: TopoVerification,
        has_ref: bool,
    ) -> TopoFormulaTerms:
        i_non_empty = 1.0 if dag.num_nodes > 0 else 0.0
        i_acyclic = 1.0 if v.acyclic >= 1.0 else 0.0
        rho_orphan = self._safe_clip01(1.0 - v.no_orphan)
        i_no_orphan = 1.0 if rho_orphan <= 1e-12 else 0.0
        delta = self._safe_clip01(v.direction_consistency)
        kappa = self._safe_clip01(v.ref_edge_f1) if has_ref else 0.0

        term_base = self.LAMBDA_BASE * i_non_empty
        term_acyclic = self.LAMBDA_ACYCLIC * i_acyclic
        term_orphan = self.LAMBDA_ORPHAN * i_no_orphan
        term_delta = self.LAMBDA_DELTA * delta
        lambda_kappa = self.LAMBDA_KAPPA if has_ref else 0.0
        term_kappa = lambda_kappa * kappa
        denom = max(
            1e-12,
            self.LAMBDA_BASE
            + self.LAMBDA_ACYCLIC
            + self.LAMBDA_ORPHAN
            + self.LAMBDA_DELTA
            + lambda_kappa,
        )
        r_topo = self._safe_clip01((term_base + term_acyclic + term_orphan + term_delta + term_kappa) / denom)
        return TopoFormulaTerms(
            lambda_base=self.LAMBDA_BASE,
            lambda_acyclic=self.LAMBDA_ACYCLIC,
            lambda_orphan=self.LAMBDA_ORPHAN,
            lambda_delta=self.LAMBDA_DELTA,
            lambda_kappa=lambda_kappa,
            indicator_non_empty=i_non_empty,
            indicator_acyclic=i_acyclic,
            indicator_no_orphan=i_no_orphan,
            rho_orphan=rho_orphan,
            delta=delta,
            kappa=kappa,
            term_base=term_base,
            term_acyclic=term_acyclic,
            term_orphan=term_orphan,
            term_delta=term_delta,
            term_kappa=term_kappa,
            denom=denom,
            r_topo=r_topo,
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
        self.last_diagnostics = []
        for idx, completion in enumerate(completions):
            text = completion_to_text(completion)
            think_text = extract_think_block(text)
            steps = extract_steps_from_answer(think_text)
            if not steps:
                rewards.append(0.0)
                continue

            ref = self._parse_ref(refs[idx] if idx < len(refs) else None)
            dag = build_dag_from_answer(think_text, reference_dag=ref)
            verification = self._compute_verification(dag, len(steps), ref)
            has_ref = ref is not None
            terms = self._compute_formula_terms(dag, verification, has_ref=has_ref)
            reward = terms.r_topo
            if self.REQUIRE_VALID_DAG and verification.valid_dag < 1.0:
                reward = 0.0
            rewards.append(reward)

            row = verification.as_dict()
            row['r_topo'] = reward
            row.update(terms.as_dict())
            diag_rows.append(row)
            self.last_diagnostics.append(row)

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

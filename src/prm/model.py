from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from src.data.build_dag import build_dag_from_answer
from src.reward.continuity_reward import ContinuityReward
from src.reward.topo_reward import TopoReward


@dataclass
class ProcessRewardResult:
    topology_reward: float
    continuity_reward: float
    dag_nodes: int
    dag_edges: int


class VerifiableProcessRewardModel:
    """Deterministic PRM over extracted dependency DAGs.

    This module computes structural process rewards by deterministic programs.
    It does not claim semantic proof verification.
    """

    def __init__(self) -> None:
        self._topology = TopoReward()
        self._continuity = ContinuityReward()

    def score_trace(self, trace: str, reference_dag: Optional[Any] = None) -> ProcessRewardResult:
        wrapped = [[{"content": f"<think>{trace}</think><answer>{{}}</answer>"}]]
        topo = float(self._topology(wrapped, reference_dag=reference_dag)[0])
        cont = float(self._continuity(wrapped)[0])
        dag = build_dag_from_answer(trace)
        return ProcessRewardResult(
            topology_reward=topo,
            continuity_reward=cont,
            dag_nodes=dag.num_nodes,
            dag_edges=dag.num_edges,
        )

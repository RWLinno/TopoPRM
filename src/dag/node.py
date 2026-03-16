from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional


class StepType(str, Enum):
    DEFINITION = "definition"
    DERIVATION = "derivation"
    COMPUTATION = "computation"
    CONCLUSION = "conclusion"
    AUXILIARY = "auxiliary"
    SUBSTITUTION = "substitution"
    CASE_ANALYSIS = "case_analysis"
    UNKNOWN = "unknown"


class LocalVerdict(str, Enum):
    CORRECT = "correct"
    INCORRECT = "incorrect"
    UNVERIFIABLE = "unverifiable"


@dataclass
class Node:
    step_id: int
    raw_text: str
    normalized_text: str = ""
    exprs: List[str] = field(default_factory=list)
    claims: List[str] = field(default_factory=list)
    step_type: StepType = StepType.DERIVATION
    local_verdict: LocalVerdict = LocalVerdict.UNVERIFIABLE
    sub_question_id: Optional[int] = None

    def to_dict(self) -> dict:
        return {
            "step_id": self.step_id,
            "raw_text": self.raw_text,
            "normalized_text": self.normalized_text,
            "exprs": self.exprs,
            "claims": self.claims,
            "step_type": self.step_type.value,
            "local_verdict": self.local_verdict.value,
            "sub_question_id": self.sub_question_id,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Node":
        return cls(
            step_id=d["step_id"],
            raw_text=d["raw_text"],
            normalized_text=d.get("normalized_text", ""),
            exprs=d.get("exprs", []),
            claims=d.get("claims", []),
            step_type=StepType(d.get("step_type", "derivation")),
            local_verdict=LocalVerdict(d.get("local_verdict", "unverifiable")),
            sub_question_id=d.get("sub_question_id"),
        )


@dataclass
class Edge:
    source: int
    target: int
    edge_type: str = "sequential"
    dep_type: str = ""
    weight: float = 0.5

    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "target": self.target,
            "edge_type": self.edge_type,
            "dep_type": self.dep_type,
            "weight": self.weight,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Edge":
        return cls(
            source=d["source"],
            target=d["target"],
            edge_type=d.get("edge_type", "sequential"),
            dep_type=d.get("dep_type", ""),
            weight=d.get("weight", 0.5),
        )

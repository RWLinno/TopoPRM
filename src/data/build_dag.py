"""Build a ReasoningDAG from a textual math answer.

Covers step extraction, expression / claim mining, step-type classification,
and rule-based (with LLM-fallback placeholder) dependency detection.

Usage::

    python -m src.data.build_dag \\
        --input_path data/processed/parsed.jsonl \\
        --output_dir  data/dag
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.dag.node import Node, StepType, LocalVerdict, Edge
from src.dag.graph import ReasoningDAG

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Sub-question marker patterns
# ---------------------------------------------------------------------------

_SUB_Q_PATTERNS: List[re.Pattern] = [
    re.compile(r"【小题(\d+)】"),
    re.compile(r"^\s*\((\d+)\)\s*"),
    re.compile(r"^\s*（(\d+)）\s*"),
    re.compile(r"^\s*第\s*(\d+)\s*[小题问]"),
]


def _detect_sub_question(text: str) -> Optional[int]:
    for pat in _SUB_Q_PATTERNS:
        m = pat.search(text)
        if m:
            return int(m.group(1))
    return None


# ---------------------------------------------------------------------------
# Step extraction
# ---------------------------------------------------------------------------

def extract_steps_from_answer(standard_answer: str) -> List[Dict[str, Any]]:
    """Split *standard_answer* into individual steps.

    Each step is a dict with ``step_id``, ``raw_text``, and an optional
    ``sub_question_id``.
    """
    lines = [l.strip() for l in standard_answer.splitlines() if l.strip()]
    if not lines:
        return []

    steps: List[Dict[str, Any]] = []
    current_sub_q: Optional[int] = None

    for idx, line in enumerate(lines):
        sq = _detect_sub_question(line)
        if sq is not None:
            current_sub_q = sq

        steps.append({
            "step_id": idx,
            "raw_text": line,
            "sub_question_id": current_sub_q,
        })

    return steps


# ---------------------------------------------------------------------------
# Expression extraction
# ---------------------------------------------------------------------------

_INLINE_MATH_RE = re.compile(
    r"\$([^$]+)\$"
    r"|"
    r"\\\((.+?)\\\)"
)

_EQUATION_RE = re.compile(
    r"[a-zA-Z\u03b1-\u03c9\u0391-\u03a9\d][a-zA-Z\u03b1-\u03c9\u0391-\u03a9\d\s+\-*/^(){}]*"
    r"[=\u2260<>\u2264\u2265\u2248]"
    r"[a-zA-Z\u03b1-\u03c9\u0391-\u03a9\d\s+\-*/^(){}]+"
)

_VAR_ASSIGN_RE = re.compile(
    r"(?:\u8bbe|\u4ee4)\s*([a-zA-Z\u03b1-\u03c9\u0391-\u03a9]\w*)\s*[=\uff1d]\s*(.+?)(?:[,\uff0c;\uff1b\u3002]|$)"
)


def extract_expressions(text: str) -> List[str]:
    """Extract mathematical expressions from *text*."""
    exprs: List[str] = []

    for m in _INLINE_MATH_RE.finditer(text):
        expr = m.group(1) or m.group(2)
        if expr:
            exprs.append(expr.strip())

    for m in _EQUATION_RE.finditer(text):
        candidate = m.group(0).strip()
        if len(candidate) >= 3 and candidate not in exprs:
            exprs.append(candidate)

    for m in _VAR_ASSIGN_RE.finditer(text):
        assign = f"{m.group(1)}={m.group(2).strip()}"
        if assign not in exprs:
            exprs.append(assign)

    return exprs


# ---------------------------------------------------------------------------
# Claim extraction
# ---------------------------------------------------------------------------

_CLAIM_PATTERNS: List[re.Pattern] = [
    re.compile(r"[A-Za-z\u03b1-\u03c9\u0391-\u03a9]+\s*[=\u2260<>\u2264\u2265\u2248]\s*\d*[A-Za-z\u03b1-\u03c9\u0391-\u03a9]+"),
    re.compile(r"[A-Z]{2}\s*[\u2225\u22a5\u2245\u223d]\s*[A-Z]{2}"),
    re.compile(r"\u2220[A-Za-z]+\s*=\s*\d+\u00b0?"),
    re.compile(r"[\u2235\u2234]\s*(.+?)(?:[,\uff0c;\uff1b\u3002]|$)"),
]


def extract_claims(text: str) -> List[str]:
    """Extract mathematical claims / relations from *text*."""
    claims: List[str] = []
    for pat in _CLAIM_PATTERNS:
        for m in pat.finditer(text):
            claim = m.group(0).strip()
            if claim and claim not in claims:
                claims.append(claim)
    return claims


# ---------------------------------------------------------------------------
# Step-type classification
# ---------------------------------------------------------------------------

_TYPE_RULES: List[Tuple[List[str], StepType]] = [
    (["\u2235", "\u5df2\u77e5", "\u7531\u9898\u610f", "\u6839\u636e\u9898\u610f", "\u9898\u76ee\u7ed9\u51fa"], StepType.DEFINITION),
    (["\u2234", "\u63a8\u5f97", "\u6240\u4ee5", "\u56e0\u6b64", "\u7531\u6b64\u53ef\u5f97", "\u5219"], StepType.DERIVATION),
    (["\u89e3\u5f97", "\u8ba1\u7b97", "\u5316\u7b80", "\u6574\u7406\u5f97"], StepType.COMPUTATION),
    (["\u6545", "\u7efc\u4e0a", "\u7efc\u4e0a\u6240\u8ff0", "\u7b54", "\u56e0\u6b64\u7b54\u6848"], StepType.CONCLUSION),
    (["\u8fde\u63a5", "\u4f5c", "\u8fc7\u70b9", "\u5ef6\u957f"], StepType.AUXILIARY),
    (["\u4ee3\u5165", "\u4ee4", "\u5c06.*\u4ee3\u5165", "\u628a.*\u4ee3\u5165"], StepType.SUBSTITUTION),
    (["\u5206\u7c7b\u8ba8\u8bba", "\u5f53.*\u65f6", "\u5206\u4e24\u79cd\u60c5\u51b5", "\u60c5\u51b5\u4e00", "\u60c5\u51b5\u4e8c"], StepType.CASE_ANALYSIS),
]


def classify_step_type(text: str) -> StepType:
    """Return the most likely :class:`StepType` for *text*."""
    for keywords, stype in _TYPE_RULES:
        for kw in keywords:
            if re.search(kw, text):
                return stype

    if re.search(r"[=\uff1d]", text) and not re.search(r"[\u2235\u2234]", text):
        return StepType.COMPUTATION

    return StepType.UNKNOWN


# ---------------------------------------------------------------------------
# Dependency-edge building
# ---------------------------------------------------------------------------

def build_dependency_edges_by_rules(
    nodes: List[Node],
) -> List[Tuple[int, int, str]]:
    """Heuristic: if *node_j* references an expression or claim that first
    appeared in an earlier *node_i*, add a dependency edge.
    """
    edges: List[Tuple[int, int, str]] = []
    expr_origin: Dict[str, int] = {}
    claim_origin: Dict[str, int] = {}

    for node in sorted(nodes, key=lambda n: n.step_id):
        for expr in node.exprs:
            if expr not in expr_origin:
                expr_origin[expr] = node.step_id

        for claim in node.claims:
            if claim not in claim_origin:
                claim_origin[claim] = node.step_id

    for node in sorted(nodes, key=lambda n: n.step_id):
        seen_sources: set[int] = set()
        for expr in node.exprs:
            src = expr_origin.get(expr)
            if src is not None and src < node.step_id and src not in seen_sources:
                edges.append((src, node.step_id, "expr_ref"))
                seen_sources.add(src)

        for claim in node.claims:
            src = claim_origin.get(claim)
            if src is not None and src < node.step_id and src not in seen_sources:
                edges.append((src, node.step_id, "claim_ref"))
                seen_sources.add(src)

        if not seen_sources and node.step_id > 0:
            if node.step_type in (StepType.DERIVATION, StepType.CONCLUSION):
                edges.append((node.step_id - 1, node.step_id, "implicit"))

    return edges


def build_dependency_edges_by_llm(
    nodes: List[Node],
    llm_client: Any = None,
) -> List[Tuple[int, int, str]]:
    """LLM-based dependency detection (placeholder).

    Falls back to rule-based detection when no *llm_client* is provided.
    """
    if llm_client is None:
        logger.debug("No LLM client -- falling back to rule-based edges")
        return build_dependency_edges_by_rules(nodes)

    logger.warning("LLM dependency detection not yet implemented; using rules")
    return build_dependency_edges_by_rules(nodes)


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def build_dag_from_answer(
    answer: str,
    problem_id: str = "",
) -> ReasoningDAG:
    """Build a complete :class:`ReasoningDAG` from a textual answer.

    Steps
    -----
    1. Split into individual steps.
    2. Extract expressions and claims per step.
    3. Classify each step's type.
    4. Add sequential edges.
    5. Add dependency edges (rule-based).
    """
    dag = ReasoningDAG(problem_id=problem_id)

    raw_steps = extract_steps_from_answer(answer)
    if not raw_steps:
        return dag

    nodes: List[Node] = []
    for s in raw_steps:
        text = s["raw_text"]
        node = Node(
            step_id=s["step_id"],
            raw_text=text,
            normalized_text=text.strip(),
            exprs=extract_expressions(text),
            claims=extract_claims(text),
            step_type=classify_step_type(text),
            local_verdict=LocalVerdict.UNVERIFIABLE,
            sub_question_id=s.get("sub_question_id"),
        )
        nodes.append(node)
        dag.add_node(node)

    dag.add_sequential_edges()

    dep_edges = build_dependency_edges_by_rules(nodes)
    for src_id, tgt_id, dep_type in dep_edges:
        dag.add_dependency_edge(src_id, tgt_id, dep_type)

    return dag


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build DAGs from parsed math-answer records."
    )
    parser.add_argument(
        "--input_path",
        type=str,
        default="data/processed/parsed.jsonl",
        help="JSONL file produced by parse_raw.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/dag",
        help="Directory to write per-record DAG JSON files.",
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    with open(args.input_path, "r", encoding="utf-8") as fin:
        for line_no, line in enumerate(fin, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                logger.warning("Line %d: bad JSON -- %s", line_no, exc)
                continue

            answer = record.get("standard_answer", "")
            rid = record.get("record_id", f"record_{line_no}")
            dag = build_dag_from_answer(answer, problem_id=rid)

            out_file = out_dir / f"{rid}.json"
            out_file.write_text(dag.to_json(), encoding="utf-8")
            count += 1

    logger.info("Built %d DAGs -> %s", count, out_dir)


if __name__ == "__main__":
    main()

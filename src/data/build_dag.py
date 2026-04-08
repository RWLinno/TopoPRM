"""Build a ReasoningDAG from textual math answers."""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.dag.graph import DOUBLE_BARRIER_EDGE, VIRTUAL_EDGE, ReasoningDAG
from src.dag.node import LocalVerdict, Node, StepType

logger = logging.getLogger(__name__)

_SUB_Q_PATTERNS: List[re.Pattern] = [
    re.compile(r"【小题(\d+)】"),
    re.compile(r"^\s*\((\d+)\)\s*"),
    re.compile(r"^\s*（(\d+)）\s*"),
    re.compile(r"^\s*第\s*(\d+)\s*[小题问]"),
]

_STEP_MARKER_RE = re.compile(
    r"^\s*(?:"
    r"(?:step|步骤)\s*\d+[:：.\-、]?"
    r"|"
    r"[（(]?\d+[)）][、.．:]?"
    r"|"
    r"(?:第\s*\d+\s*步)[:：]?"
    r"|"
    r"[一二三四五六七八九十]+[、.．:]"
    r")\s*",
    re.IGNORECASE,
)
_INLINE_STEP_SPLIT_RE = re.compile(
    r"(?=(?:^|[\s。；;])(?:step\s*\d+|步骤\s*\d+|[（(]?\d+[)）][、.．:]?|第\s*\d+\s*步[:：]?))",
    re.IGNORECASE,
)
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[。！？!?;；])\s+")

_LATEX_CMD_RE = re.compile(r"\\[A-Za-z]+")
_MATH_WS_RE = re.compile(r"\s+")
_NON_TEXT_TOKEN = re.compile(r"[^\w\u4e00-\u9fff]+")
_TOKEN_VAR_RE = re.compile(r"[A-Za-z\u03b1-\u03c9\u0391-\u03a9][A-Za-z0-9_]*")

_VAR_STOPWORDS = {
    "step", "steps", "let", "given", "thus", "therefore", "hence", "then",
    "answer", "proof", "case", "assume", "suppose", "show",
}

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
    r"(?:\u8bbe|\u4ee4|let)\s*([a-zA-Z\u03b1-\u03c9\u0391-\u03a9]\w*)\s*[=\uff1d]\s*(.+?)(?:[,\uff0c;\uff1b\u3002]|$)",
    re.IGNORECASE,
)

_CLAIM_PATTERNS: List[re.Pattern] = [
    re.compile(r"[A-Za-z\u03b1-\u03c9\u0391-\u03a9]+\s*[=\u2260<>\u2264\u2265\u2248]\s*\d*[A-Za-z\u03b1-\u03c9\u0391-\u03a9]+"),
    re.compile(r"[A-Z]{2}\s*[\u2225\u22a5\u2245\u223d]\s*[A-Z]{2}"),
    re.compile(r"\u2220[A-Za-z]+\s*=\s*\d+\u00b0?"),
    re.compile(r"[\u2235\u2234]\s*(.+?)(?:[,\uff0c;\uff1b\u3002]|$)"),
]

_CLAIM_VERB_HINTS = (
    "是", "为", "等于", "得到", "可得", "推出", "所以", "因此", "故", "则", "说明", "成立", "不成立",
    "平行", "垂直", "相等", "同余", "大于", "小于", "不少于", "不大于",
)

_TYPE_RULES: List[Tuple[List[str], StepType]] = [
    (["\u2235", "\u5df2\u77e5", "\u7531\u9898\u610f", "\u6839\u636e\u9898\u610f", "\u9898\u76ee\u7ed9\u51fa"], StepType.DEFINITION),
    (["\u2234", "\u63a8\u5f97", "\u6240\u4ee5", "\u56e0\u6b64", "\u7531\u6b64\u53ef\u5f97", "\u5219"], StepType.DERIVATION),
    (["\u89e3\u5f97", "\u8ba1\u7b97", "\u5316\u7b80", "\u6574\u7406\u5f97"], StepType.COMPUTATION),
    (["\u6545", "\u7efc\u4e0a", "\u7efc\u4e0a\u6240\u8ff0", "\u7b54", "\u56e0\u6b64\u7b54\u6848"], StepType.CONCLUSION),
    (["\u8fde\u63a5", "\u4f5c", "\u8fc7\u70b9", "\u5ef6\u957f"], StepType.AUXILIARY),
    (["\u4ee3\u5165", "\u4ee4", "\u5c06.*\u4ee3\u5165", "\u628a.*\u4ee3\u5165"], StepType.SUBSTITUTION),
    (["\u5206\u7c7b\u8ba8\u8bba", "\u5f53.*\u65f6", "\u5206\u4e24\u79cd\u60c5\u51b5", "\u60c5\u51b5\u4e00", "\u60c5\u51b5\u4e8c"], StepType.CASE_ANALYSIS),
]

_ENABLE_SEQ_WEAK_EDGE = (os.environ.get("TOPO_ENABLE_SEQUENTIAL_WEAK_EDGE", "1") or "1") != "0"
_SEQ_WEAK_EDGE_MODE = (os.environ.get("TOPO_SEQ_WEAK_EDGE_MODE", "adaptive") or "adaptive").lower()


@dataclass
class ParsedStep:
    step_id: int
    raw_text: str
    normalized_text: str
    sub_question_id: Optional[int]
    exprs: List[str]
    claims: List[str]
    claim_keys: List[str]
    variables: List[str]
    step_type: StepType


@dataclass
class EdgeEvidence:
    source: int
    target: int
    edge_type: str
    dep_type: str
    evidence: str


def _detect_sub_question(text: str) -> Optional[int]:
    for pat in _SUB_Q_PATTERNS:
        m = pat.search(text)
        if m:
            return int(m.group(1))
    return None


def _normalize_answer_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("<br>", "\n").replace("<br/>", "\n").replace("<br />", "\n")
    return text


def _split_inline_steps(line: str) -> List[str]:
    line = line.strip()
    if not line:
        return []
    parts = [p.strip(" \t;；") for p in _INLINE_STEP_SPLIT_RE.split(line) if p and p.strip(" \t;；")]
    if len(parts) == 1 and len(line) > 140 and ("；" in line or ";" in line):
        punct_parts = [p.strip() for p in re.split(r"[；;]+", line) if p.strip()]
        if len(punct_parts) > 1:
            return punct_parts
    return parts


def _looks_incomplete_fragment(text: str) -> bool:
    t = text.strip()
    if not t:
        return True
    if t.endswith((":", "：", ",", "，", ";", "；")):
        return True
    # Typical heading-like fragments that should not become standalone nodes.
    if re.match(r"^(思路|分析|解题思路|设|已知|证明|结论)\s*[:：]?$", t, re.IGNORECASE):
        return True
    return False


def _split_sentences(text: str) -> List[str]:
    raw = [s.strip() for s in _SENTENCE_SPLIT_RE.split(text) if s.strip()]
    if not raw:
        return []
    out: List[str] = []
    for seg in raw:
        parts = [p.strip() for p in re.split(r"[。！？!?；;]+", seg) if p.strip()]
        if parts:
            out.extend(parts)
        else:
            out.append(seg)
    return out


def _is_complete_claim_sentence(text: str) -> bool:
    t = text.strip()
    if re.search(r"[=\u2260<>\u2264\u2265\u2248\u2225\u22a5\u2245\u223d]", t) and len(t) >= 3:
        return True
    if _looks_incomplete_fragment(t):
        return False
    if len(t) < 6:
        return False
    if re.search(r"[=\u2260<>\u2264\u2265\u2248]", t):
        return True
    return any(hint in t for hint in _CLAIM_VERB_HINTS)


def _normalize_claim_sentence(text: str) -> str:
    t = unicodedata.normalize("NFKC", text).strip()
    return t.lower()


def extract_steps_from_answer(standard_answer: str) -> List[Dict[str, Any]]:
    normalized = _normalize_answer_text(standard_answer)
    raw_lines = [l.strip() for l in normalized.splitlines() if l.strip()]
    if not raw_lines:
        return []

    lines: List[str] = []
    for line in raw_lines:
        lines.extend(_split_inline_steps(line))

    steps: List[Dict[str, Any]] = []
    current_sub_q: Optional[int] = None
    for line in lines:
        line = _STEP_MARKER_RE.sub("", line).strip()
        if not line:
            continue
        if _looks_incomplete_fragment(line):
            continue
        sq = _detect_sub_question(line)
        if sq is not None:
            current_sub_q = sq
        steps.append(
            {
                "step_id": len(steps),
                "raw_text": line,
                "sub_question_id": current_sub_q,
            }
        )
    return steps


def canonicalize_expression(expr: str) -> str:
    expr = unicodedata.normalize("NFKC", expr).strip()
    expr = _LATEX_CMD_RE.sub("", expr)
    expr = expr.replace("×", "*").replace("÷", "/").replace("−", "-")
    expr = expr.replace("＝", "=").replace("≤", "<=").replace("≥", ">=")
    expr = _MATH_WS_RE.sub("", expr)
    expr = expr.strip("，,。.;；:：")
    if expr.startswith("(") and expr.endswith(")") and len(expr) > 2:
        expr = expr[1:-1]
    return expr.lower()


def extract_variables(text: str) -> List[str]:
    normalized = unicodedata.normalize("NFKC", text)
    vars_found: List[str] = []
    for tok in _TOKEN_VAR_RE.findall(normalized):
        t = tok.lower()
        if t in _VAR_STOPWORDS:
            continue
        if len(t) > 1 and t.isalpha() and t not in {"xy", "yz", "ab", "bc", "cd", "sin", "cos", "tan", "log"}:
            continue
        vars_found.append(t)
    return sorted(set(vars_found))


def extract_expressions(text: str) -> List[str]:
    exprs_raw: List[str] = []
    for m in _INLINE_MATH_RE.finditer(text):
        expr = m.group(1) or m.group(2)
        if expr:
            exprs_raw.append(expr.strip())
    for m in _EQUATION_RE.finditer(text):
        candidate = m.group(0).strip()
        if len(candidate) >= 3:
            exprs_raw.append(candidate)
    for m in _VAR_ASSIGN_RE.finditer(text):
        exprs_raw.append(f"{m.group(1)}={m.group(2).strip()}")

    exprs: List[str] = []
    for item in exprs_raw:
        canonical = canonicalize_expression(item)
        if len(canonical) >= 2 and canonical not in exprs:
            exprs.append(canonical)
    return exprs


def extract_claims(text: str) -> List[str]:
    """Extract sentence-level claims for node display and structural reasoning.

    We intentionally keep full claim sentences (instead of atom-level snippets)
    to avoid incomplete fragments such as '所有有两种运输方案：'.
    """
    claims: List[str] = []
    for sent in _split_sentences(unicodedata.normalize("NFKC", text)):
        normalized = _normalize_claim_sentence(sent)
        if _is_complete_claim_sentence(normalized) and normalized not in claims:
            claims.append(normalized)
    return claims


def extract_claim_keys(text: str) -> List[str]:
    """Extract canonical claim keys for dependency matching."""
    claims: List[str] = []
    for pat in _CLAIM_PATTERNS:
        for m in pat.finditer(text):
            claim = canonicalize_expression(m.group(0).strip())
            if claim and claim not in claims:
                claims.append(claim)
    return claims


def classify_step_type(text: str) -> StepType:
    for keywords, stype in _TYPE_RULES:
        for kw in keywords:
            if re.search(kw, text):
                return stype
    if re.search(r"[=\uff1d]", text) and not re.search(r"[\u2235\u2234]", text):
        return StepType.COMPUTATION
    return StepType.UNKNOWN


def _overlap_ratio(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    ta = set(filter(None, _NON_TEXT_TOKEN.split(a)))
    tb = set(filter(None, _NON_TEXT_TOKEN.split(b)))
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / max(len(ta), len(tb))


def build_dependency_edges_by_rules(
    steps: List[ParsedStep],
) -> Tuple[List[Tuple[int, int, str, str]], List[EdgeEvidence]]:
    edges: List[Tuple[int, int, str, str]] = []
    evidences: List[EdgeEvidence] = []

    expr_origin: Dict[str, int] = {}
    claim_origin: Dict[str, int] = {}
    var_origin: Dict[str, int] = {}

    for step in sorted(steps, key=lambda n: n.step_id):
        for expr in step.exprs:
            expr_origin.setdefault(expr, step.step_id)
        for claim in step.claim_keys:
            claim_origin.setdefault(claim, step.step_id)
        for var in step.variables:
            var_origin.setdefault(var, step.step_id)

    for step in sorted(steps, key=lambda n: n.step_id):
        seen_sources: set[int] = set()

        for expr in step.exprs:
            src = expr_origin.get(expr)
            if src is not None and src < step.step_id and src not in seen_sources:
                edges.append((src, step.step_id, VIRTUAL_EDGE, "expr_ref"))
                evidences.append(EdgeEvidence(src, step.step_id, VIRTUAL_EDGE, "expr_ref", f"expr={expr}"))
                seen_sources.add(src)
                continue
            for prev_expr, prev_src in expr_origin.items():
                if prev_src >= step.step_id or prev_src in seen_sources:
                    continue
                if _overlap_ratio(expr, prev_expr) >= 0.8:
                    edges.append((prev_src, step.step_id, VIRTUAL_EDGE, "expr_overlap"))
                    evidences.append(
                        EdgeEvidence(prev_src, step.step_id, VIRTUAL_EDGE, "expr_overlap", f"{prev_expr}->{expr}")
                    )
                    seen_sources.add(prev_src)
                    break

        for claim in step.claim_keys:
            src = claim_origin.get(claim)
            if src is not None and src < step.step_id and src not in seen_sources:
                edges.append((src, step.step_id, VIRTUAL_EDGE, "claim_ref"))
                evidences.append(EdgeEvidence(src, step.step_id, VIRTUAL_EDGE, "claim_ref", f"claim={claim}"))
                seen_sources.add(src)

        for var in step.variables:
            src = var_origin.get(var)
            if src is not None and src < step.step_id and src not in seen_sources:
                edges.append((src, step.step_id, VIRTUAL_EDGE, "var_ref"))
                evidences.append(EdgeEvidence(src, step.step_id, VIRTUAL_EDGE, "var_ref", f"var={var}"))
                seen_sources.add(src)

        if not seen_sources and step.step_id > 0 and step.step_type in (StepType.DERIVATION, StepType.CONCLUSION):
            edges.append((step.step_id - 1, step.step_id, DOUBLE_BARRIER_EDGE, "implicit_block"))
            evidences.append(
                EdgeEvidence(step.step_id - 1, step.step_id, DOUBLE_BARRIER_EDGE, "implicit_block", "fallback")
            )

    return edges, evidences


def build_dependency_edges_by_llm(
    nodes: List[ParsedStep],
    llm_client: Any = None,
) -> Tuple[List[Tuple[int, int, str, str]], List[EdgeEvidence]]:
    if llm_client is None:
        logger.debug("No LLM client -- falling back to rule-based edges")
        return build_dependency_edges_by_rules(nodes)
    logger.warning("LLM dependency detection not yet implemented; using rules")
    return build_dependency_edges_by_rules(nodes)


def _should_add_seq_edge(dag: ReasoningDAG, src_id: int, tgt_id: int) -> bool:
    mode = _SEQ_WEAK_EDGE_MODE
    if mode == "off":
        return False
    if mode == "full":
        return True
    # adaptive (default): only add weak sequential edge when target has
    # no incoming dependency/barrier edge yet, reducing chain-like domination.
    for u, v, data in dag.graph.in_edges(tgt_id, data=True):
        _ = (u, v)
        et = data.get("edge_type", "")
        if dag.is_virtual_edge(et) or et == DOUBLE_BARRIER_EDGE:
            return False
    # Also avoid clutter when src already has a strong outgoing edge.
    for u, v, data in dag.graph.out_edges(src_id, data=True):
        _ = (u, v)
        et = data.get("edge_type", "")
        if dag.is_virtual_edge(et):
            return False
    return True


def _add_sequential_weak_edges(dag: ReasoningDAG) -> int:
    ids = sorted(dag.nodes)
    added = 0
    for a, b in zip(ids, ids[1:]):
        if _should_add_seq_edge(dag, a, b) and not dag.graph.has_edge(a, b):
            dag.graph.add_edge(
                a,
                b,
                weight=0.3,
                edge_type="solid_edge",
                dep_type="order",
            )
            added += 1
    return added


def _parse_reference_dag(raw_ref: Any) -> Optional[ReasoningDAG]:
    if raw_ref is None:
        return None
    try:
        if isinstance(raw_ref, ReasoningDAG):
            return raw_ref
        if isinstance(raw_ref, dict):
            return ReasoningDAG.from_dict(raw_ref)
        if isinstance(raw_ref, str) and raw_ref.strip():
            return ReasoningDAG.from_json(raw_ref)
    except Exception:
        return None
    return None


def _fallback_verdict(step: ParsedStep, has_virtual_in: bool, has_virtual_out: bool) -> LocalVerdict:
    text = step.normalized_text
    if re.search(r"错误|不成立|矛盾|有误|invalid|wrong", text, re.IGNORECASE):
        return LocalVerdict.INCORRECT
    if step.step_type == StepType.CONCLUSION and not has_virtual_in:
        return LocalVerdict.INCORRECT
    if step.exprs or step.claims or has_virtual_in or has_virtual_out:
        return LocalVerdict.CORRECT
    return LocalVerdict.UNVERIFIABLE


def _assign_hybrid_verdicts(
    dag: ReasoningDAG,
    parsed_steps: List[ParsedStep],
    ref_dag: Optional[ReasoningDAG],
) -> None:
    ref_map: Dict[int, LocalVerdict] = {}
    if ref_dag is not None:
        for sid, node in ref_dag.nodes.items():
            ref_map[sid] = node.local_verdict

    by_sid = {s.step_id: s for s in parsed_steps}
    for sid, node in dag.nodes.items():
        if sid in ref_map:
            node.local_verdict = ref_map[sid]
            continue
        has_virtual_in = any(
            dag.is_virtual_edge(data.get("edge_type", ""))
            for _, _, data in dag.graph.in_edges(sid, data=True)
        )
        has_virtual_out = any(
            dag.is_virtual_edge(data.get("edge_type", ""))
            for _, _, data in dag.graph.out_edges(sid, data=True)
        )
        step = by_sid.get(sid)
        if step is None:
            node.local_verdict = LocalVerdict.UNVERIFIABLE
        else:
            node.local_verdict = _fallback_verdict(step, has_virtual_in, has_virtual_out)


def parse_answer_to_dag_debug(
    answer: str,
    problem_id: str = "",
    reference_dag: Optional[Any] = None,
) -> Tuple[ReasoningDAG, Dict[str, Any]]:
    dag = ReasoningDAG(problem_id=problem_id)
    debug: Dict[str, Any] = {
        "problem_id": problem_id,
        "raw_answer": answer,
        "steps": [],
        "edges": [],
    }
    raw_steps = extract_steps_from_answer(answer)
    if not raw_steps:
        return dag, debug

    parsed_steps: List[ParsedStep] = []
    parsed_ref_dag = _parse_reference_dag(reference_dag)
    for s in raw_steps:
        text = s["raw_text"]
        norm = unicodedata.normalize("NFKC", text).strip()
        exprs = extract_expressions(text)
        claims = extract_claims(text)
        claim_keys = extract_claim_keys(text)
        variables = extract_variables(f"{norm} {' '.join(exprs)}")
        stype = classify_step_type(text)

        parsed = ParsedStep(
            step_id=s["step_id"],
            raw_text=text,
            normalized_text=norm,
            sub_question_id=s.get("sub_question_id"),
            exprs=exprs,
            claims=claims,
            claim_keys=claim_keys,
            variables=variables,
            step_type=stype,
        )
        parsed_steps.append(parsed)

        dag.add_node(
            Node(
                step_id=parsed.step_id,
                raw_text=parsed.raw_text,
                normalized_text=parsed.normalized_text,
                exprs=parsed.exprs,
                claims=parsed.claims,
                step_type=parsed.step_type,
                local_verdict=LocalVerdict.UNVERIFIABLE,
                sub_question_id=parsed.sub_question_id,
            )
        )
        debug["steps"].append(
            {
                "step_id": parsed.step_id,
                "raw_text": parsed.raw_text,
                "normalized_text": parsed.normalized_text,
                "exprs": parsed.exprs,
                "claims": parsed.claims,
                "claim_keys": parsed.claim_keys,
                "variables": parsed.variables,
                "step_type": parsed.step_type.value,
                "sub_question_id": parsed.sub_question_id,
            }
        )

    dep_edges, evidences = build_dependency_edges_by_rules(parsed_steps)
    for src_id, tgt_id, edge_type, dep_type in dep_edges:
        if edge_type == VIRTUAL_EDGE:
            dag.add_dependency_edge(src_id, tgt_id, dep_type)
        else:
            dag.add_implicit_barrier_edge(src_id, tgt_id)

    seq_edges_added = 0
    if _ENABLE_SEQ_WEAK_EDGE:
        seq_edges_added = _add_sequential_weak_edges(dag)

    _assign_hybrid_verdicts(dag, parsed_steps, parsed_ref_dag)

    debug["edges"] = [
        {
            "source": e.source,
            "target": e.target,
            "edge_type": e.edge_type,
            "dep_type": e.dep_type,
            "evidence": e.evidence,
        }
        for e in evidences
    ]
    edge_source_stats: Dict[str, int] = {}
    for _, _, d in dag.graph.edges(data=True):
        dep_type = d.get("dep_type", "unknown")
        edge_source_stats[dep_type] = edge_source_stats.get(dep_type, 0) + 1
    verdict_stats: Dict[str, int] = {}
    for n in dag.nodes.values():
        k = n.local_verdict.value
        verdict_stats[k] = verdict_stats.get(k, 0) + 1

    debug["summary"] = {
        "num_steps": len(parsed_steps),
        "num_nodes": dag.num_nodes,
        "num_edges": dag.num_edges,
        "enable_sequential_weak_edge": _ENABLE_SEQ_WEAK_EDGE,
        "sequential_mode": _SEQ_WEAK_EDGE_MODE,
        "seq_edges_added": seq_edges_added,
        "edge_source_stats": edge_source_stats,
        "verdict_stats": verdict_stats,
    }
    return dag, debug


def build_dag_from_answer(
    answer: str,
    problem_id: str = "",
    reference_dag: Optional[Any] = None,
) -> ReasoningDAG:
    dag, _ = parse_answer_to_dag_debug(
        answer=answer,
        problem_id=problem_id,
        reference_dag=reference_dag,
    )
    return dag


def main() -> None:
    parser = argparse.ArgumentParser(description="Build DAGs from parsed math-answer records.")
    parser.add_argument("--input_path", type=str, default="data/processed/parsed.jsonl")
    parser.add_argument("--output_dir", type=str, default="data/dag")
    parser.add_argument("--log_level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
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

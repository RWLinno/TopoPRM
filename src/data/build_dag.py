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
_LOCAL_LLM_CLIENT: Any = None
_LOCAL_LLM_LOAD_ATTEMPTED = False


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return raw not in {"0", "false", "False", "no", "NO"}


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default

_SUB_Q_PATTERNS: List[re.Pattern] = [
    re.compile(r"【小题(\d+)】"),
    re.compile(r"^\s*\((\d+)\)\s*"),
    re.compile(r"^\s*（(\d+)）\s*"),
    re.compile(r"^\s*第\s*(\d+)\s*[小题问]"),
]
_SUB_Q_STRIP_PATTERNS: List[re.Pattern] = [
    re.compile(r"^\s*【小题\d+】\s*"),
    re.compile(r"^\s*[（(]?\d+[)）]\s*"),
    re.compile(r"^\s*第\s*\d+\s*[小题问]\s*"),
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
    r"|"
    r"\*\*(?:step|steps?)\s*\d+\*\*[:.\-]?"
    r"|"
    r"(?:first|second|third|fourth|fifth|next|then|finally)[,:]"
    r")\s*",
    re.IGNORECASE,
)
_INLINE_STEP_SPLIT_RE = re.compile(
    r"(?=(?:^|[\s。；;.])(?:step\s*\d+|步骤\s*\d+|[（(]?\d+[)）][、.．:]?|第\s*\d+\s*步[:：]?))",
    re.IGNORECASE,
)
# Lines that are pure formatting / delimiters and should not become DAG nodes.
# Filtering is opt-in via TOPO_DAG_FILTER_FORMATTING=1 to preserve legacy behaviour.
_FORMATTING_NOISE_RE = re.compile(
    r"^(?:"
    r"</?think>"                                 # think tags
    r"|\\\[|\\\]|\\\(|\\\)"                       # LaTeX delimiters
    r"|\\boxed\s*\{?"                              # \boxed
    r"|\\text\s*\{[^}]*\}\s*"                     # \text{...}
    r"|\*+\s*final\s+answer\s*[:：]?\s*\*+"      # **Final Answer:**
    r"|\*+[^*]+\*+\s*[:：]?$"                     # bold-only headings
    r"|[-=*_]{3,}"                                 # markdown rules
    r"|#{1,6}\s+.*"                                # markdown headings
    r"|[\\\[\]\{\}\(\)\$]+"                       # symbol-only lines
    r"|\d+\s*\\?\s*\.?\s*\*+\s*[A-Za-z][^*]*\*+\s*[:：]?\s*$"   # "1. **Total Eggs Laid:**"
    r")\s*$",
    re.IGNORECASE,
)
_EXTRA_STEP_MARKER_RE = re.compile(
    r"^\s*(?:"
    r"(?:so|next|then|finally|after that|therefore|thus|hence)\s*[:：,\-]?"
    r"|"
    r"(?:step)\s*[:：]"
    r"|"
    r"(?:\-\s+|\*\s+)"
    r")\s*",
    re.IGNORECASE,
)
_INLINE_EXTRA_STEP_SPLIT_RE = re.compile(
    r"(?=(?:\b(?:first|second|third|next|then|so|finally|therefore|thus|hence)\b\s*[:,]?))",
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
_LATEX_EXPR_RE = re.compile(
    r"\\(?:frac|sqrt|sum|prod|int|cdot|times|leq|geq|neq|pm|mp)\b(?:\{[^{}]{1,64}\}){0,3}",
    re.IGNORECASE,
)
# GSM8K-style "<<expr=result>>" macro markers.
_GSM8K_MACRO_RE = re.compile(r"<<\s*([^<>]+?)\s*=\s*([^<>]+?)\s*>>")
_EQUATION_RE = re.compile(
    r"[a-zA-Z\u03b1-\u03c9\u0391-\u03a9\d][a-zA-Z\u03b1-\u03c9\u0391-\u03a9\d\s+\-*/^(){}]*"
    r"[=\u2260<>\u2264\u2265\u2248]"
    r"[a-zA-Z\u03b1-\u03c9\u0391-\u03a9\d\s+\-*/^(){}]+"
)
# "... = 72" / "= 72 clips" tail-result pattern used heavily by GSM8K.
_TAIL_RESULT_RE = re.compile(r"=\s*(-?\d+(?:\.\d+)?)")
# Quantity mentions: "8 purple flowers", "72 clips". Captures <number, noun>
# so that later steps referring to the same quantity trigger a virtual edge.
_QUANTITY_MENTION_RE = re.compile(
    r"(?<!\d)(-?\d+(?:\.\d+)?)\s+([a-zA-Z][a-zA-Z\u00c0-\u024f\-']{2,20})"
)
# Short list of English noun fragments that are too generic to form a
# distinctive quantity reference. Anything else is kept as-is.
_QUANTITY_NOUN_STOPSET = {
    "the", "and", "but", "for", "with", "from", "into", "that", "this",
    "hours", "hour", "minutes", "minute", "seconds", "second", "days",
    "day", "times", "time", "years", "year", "percent", "dollars",
}
_EXPR_OPERATOR_RE = re.compile(r"[=\u2260<>\u2264\u2265\u2248+\-*/^]|∥|⊥")
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
    # English claim verb hints
    "equals", "equal to", "we get", "we obtain", "we have", "we find",
    "therefore", "thus", "hence", "so ", "it follows", "implies",
    "is equal", "is greater", "is less", "divides", "is divisible",
    "is parallel", "is perpendicular", "is congruent", "is similar",
    "satisfies", "yields", "gives us", "results in",
)

_TYPE_RULES: List[Tuple[List[str], StepType]] = [
    # Chinese
    (["\u2235", "\u5df2\u77e5", "\u7531\u9898\u610f", "\u6839\u636e\u9898\u610f", "\u9898\u76ee\u7ed9\u51fa"], StepType.DEFINITION),
    (["\u2234", "\u63a8\u5f97", "\u6240\u4ee5", "\u56e0\u6b64", "\u7531\u6b64\u53ef\u5f97", "\u5219"], StepType.DERIVATION),
    (["\u89e3\u5f97", "\u8ba1\u7b97", "\u5316\u7b80", "\u6574\u7406\u5f97"], StepType.COMPUTATION),
    (["\u6545", "\u7efc\u4e0a", "\u7efc\u4e0a\u6240\u8ff0", "\u7b54", "\u56e0\u6b64\u7b54\u6848"], StepType.CONCLUSION),
    (["\u8fde\u63a5", "\u4f5c", "\u8fc7\u70b9", "\u5ef6\u957f"], StepType.AUXILIARY),
    (["\u4ee3\u5165", "\u4ee4", "\u5c06.*\u4ee3\u5165", "\u628a.*\u4ee3\u5165"], StepType.SUBSTITUTION),
    (["\u5206\u7c7b\u8ba8\u8bba", "\u5f53.*\u65f6", "\u5206\u4e24\u79cd\u60c5\u51b5", "\u60c5\u51b5\u4e00", "\u60c5\u51b5\u4e8c"], StepType.CASE_ANALYSIS),
    # English
    (["given that", "we know", "by assumption", "let ", "suppose", "assume"], StepType.DEFINITION),
    (["therefore", "thus", "hence", "it follows", "we deduce", "this gives", "implies that", "so we"], StepType.DERIVATION),
    (["computing", "calculating", "simplif", "expanding", "substitut", "evaluat", "we compute"], StepType.COMPUTATION),
    (["the answer is", "in conclusion", "finally", "the final answer", "boxed{", "\\boxed"], StepType.CONCLUSION),
    (["construct", "draw ", "extend", "connect"], StepType.AUXILIARY),
    (["substitut", "plug", "replacing", "putting.*into"], StepType.SUBSTITUTION),
    (["case 1", "case 2", "case i", "case ii", "if.*then", "consider the case", "without loss of generality"], StepType.CASE_ANALYSIS),
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
    if len(parts) <= 1 and _dag_extra_step_markers_enabled():
        extra = [p.strip(" \t;；") for p in _INLINE_EXTRA_STEP_SPLIT_RE.split(line) if p and p.strip(" \t;；")]
        if len(extra) > 1:
            return extra
    return parts


def _strip_sub_question_prefix(text: str) -> str:
    out = text
    for pat in _SUB_Q_STRIP_PATTERNS:
        out = pat.sub("", out)
    return out.strip()


def _looks_incomplete_fragment(text: str) -> bool:
    t = text.strip()
    if not t:
        return True
    if t.endswith((":", "：", ",", "，", ";", "；")):
        return True
    if re.match(r"^(思路|分析|解题思路|设|已知|证明|结论)\s*[:：]?$", t, re.IGNORECASE):
        return True
    if re.match(r"^(solution|proof|approach|strategy|method|answer|note)\s*[:.]?$", t, re.IGNORECASE):
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
        if _dag_extra_step_markers_enabled():
            line = _EXTRA_STEP_MARKER_RE.sub("", line).strip()
        if not line:
            continue
        sq = _detect_sub_question(line)
        if sq is not None:
            current_sub_q = sq
        line = _strip_sub_question_prefix(line)
        if not line:
            continue
        if _looks_incomplete_fragment(line):
            continue
        if _dag_filter_formatting_enabled() and _is_formatting_noise(line):
            continue
        steps.append(
            {
                "step_id": len(steps),
                "raw_text": line,
                "sub_question_id": current_sub_q,
            }
        )

    # P4: sentence-level fallback when no explicit step markers were found.
    # On natural-language CoT traces (no "Step N:", no bullets, no numbered
    # lists), the above loop often produces 0 or 1 steps because
    # _STEP_MARKER_RE strips the only content.  When enabled, we re-segment
    # the original text on sentence boundaries so the DAG extractor can
    # still build a multi-node graph with meaningful q_topo.
    if len(steps) <= 1 and _dag_sentence_fallback_enabled():
        steps = _sentence_fallback_split(normalized)

    return steps


def _dag_sentence_fallback_enabled() -> bool:
    """Check whether the P4 sentence-fallback flag is active."""
    return _env_bool("TOPO_DAG_SENTENCE_FALLBACK", False)


def _dag_extra_step_markers_enabled() -> bool:
    """Enable extended English step-marker normalization (R1/QwQ style)."""
    return _env_bool("TOPO_DAG_EXTRA_STEP_MARKERS", False)


def _dag_latex_expr_enabled() -> bool:
    """Enable richer LaTeX expression extraction for competition math."""
    return _env_bool("TOPO_DAG_LATEX_EXPR", False)


def _dag_barrier_strict_enabled() -> bool:
    """Enable strict implicit-block fallback (reduce fallback domination)."""
    return _env_bool("TOPO_DAG_BARRIER_STRICT", False)


def _dag_filter_formatting_enabled() -> bool:
    """Filter LaTeX delimiters / </think> / markdown chrome out of step list."""
    return _env_bool("TOPO_DAG_FILTER_FORMATTING", False)


def _dag_seq_when_no_dep_only() -> bool:
    """Add a sequential weak edge ONLY if neither side has any structural edge."""
    return _env_bool("TOPO_DAG_SEQ_WHEN_NO_DEP_ONLY", False)


def _is_formatting_noise(line: str) -> bool:
    t = line.strip()
    if not t:
        return True
    if _FORMATTING_NOISE_RE.match(t):
        return True
    # purely-symbolic single token with no alphabetic char
    if len(t) <= 3 and not re.search(r"[A-Za-z\u4e00-\u9fff]", t):
        return True
    return False


_SENTENCE_SPLIT_RE = re.compile(r'(?<=[。.!?！？\n])\s*')


def _sentence_fallback_split(text: str) -> List[Dict[str, Any]]:
    """Split text on sentence boundaries with a minimum length filter."""
    min_len = _env_int("TOPO_DAG_SENTENCE_MIN_LEN", 20)

    raw_sents = _SENTENCE_SPLIT_RE.split(text)
    steps: List[Dict[str, Any]] = []
    buf = ""
    for sent in raw_sents:
        sent = sent.strip()
        if not sent:
            continue
        buf += (" " if buf else "") + sent
        if len(buf) >= min_len:
            steps.append({
                "step_id": len(steps),
                "raw_text": buf,
                "sub_question_id": None,
                "fallback": True,
            })
            buf = ""
    # Flush remaining buffer
    if buf and len(buf) >= min_len // 2:
        steps.append({
            "step_id": len(steps),
            "raw_text": buf,
            "sub_question_id": None,
            "fallback": True,
        })
    return steps if len(steps) > 1 else []


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


def _is_informative_expression(expr: str) -> bool:
    e = expr.strip()
    if len(e) < 2:
        return False
    if re.fullmatch(r"[a-z\u03b1-\u03c9]", e):
        return False
    # "72_clips", "8_purple-flowers" quantity mentions: accept outright.
    if re.fullmatch(r"-?\d+(?:\.\d+)?_[a-z\u00c0-\u024f][a-z\u00c0-\u024f\-']+", e):
        return True
    # Keep distinctive numeric results (>= 2 digits). Drops "0".."9" which are
    # too ambiguous to indicate reuse.
    if re.fullmatch(r"-?\d+(?:\.\d+)?", e):
        return len(e.lstrip("-").split(".")[0]) >= 2
    if _EXPR_OPERATOR_RE.search(e):
        return True
    if "(" in e and ")" in e and len(e) >= 5:
        return True
    alpha_cnt = len(re.findall(r"[a-z\u03b1-\u03c9]", e))
    return alpha_cnt >= 2 and len(e) >= 5


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
    if _dag_latex_expr_enabled():
        for m in _LATEX_EXPR_RE.finditer(text):
            exprs_raw.append(m.group(0).strip())
    # GSM8K macro: <<48/2=24>> → keep both the equation and its numeric result.
    for m in _GSM8K_MACRO_RE.finditer(text):
        lhs = m.group(1).strip()
        rhs = m.group(2).strip()
        if lhs:
            exprs_raw.append(f"{lhs}={rhs}")
        if rhs:
            exprs_raw.append(rhs)
    for m in _EQUATION_RE.finditer(text):
        candidate = m.group(0).strip()
        if len(candidate) >= 3:
            exprs_raw.append(candidate)
    for m in _VAR_ASSIGN_RE.finditer(text):
        exprs_raw.append(f"{m.group(1)}={m.group(2).strip()}")
    # "= 72" tail results — keep only distinctive numbers (>= 2 digits) so the
    # expression index captures chained numeric reuse without flooding on 0/1.
    for m in _TAIL_RESULT_RE.finditer(text):
        val = m.group(1)
        if val and len(val.lstrip("-").split(".")[0]) >= 2:
            exprs_raw.append(val)

    # "72 clips", "8 purple flowers" — treat <number, noun> as a reusable
    # quantity reference so subsequent steps mentioning the same quantity
    # become virtual-edge predecessors.
    for m in _QUANTITY_MENTION_RE.finditer(text):
        num, noun = m.group(1), m.group(2).lower()
        if noun in _QUANTITY_NOUN_STOPSET:
            continue
        if len(num.lstrip("-").split(".")[0]) < 2:
            # skip single-digit-number pairings unless the noun is itself
            # non-trivial (it usually is, but be conservative).
            if len(noun) < 5:
                continue
        exprs_raw.append(f"{num}_{noun}")

    exprs: List[str] = []
    for item in exprs_raw:
        canonical = canonicalize_expression(item)
        if _is_informative_expression(canonical) and canonical not in exprs:
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
            if _is_informative_expression(claim) and claim not in claims:
                claims.append(claim)
    return claims


def _same_sub_question(a: Optional[int], b: Optional[int]) -> bool:
    return a == b


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

    expr_origin: Dict[Tuple[Optional[int], str], int] = {}
    claim_origin: Dict[Tuple[Optional[int], str], int] = {}
    var_origin: Dict[Tuple[Optional[int], str], int] = {}
    by_id: Dict[int, ParsedStep] = {s.step_id: s for s in steps}

    for step in sorted(steps, key=lambda n: n.step_id):
        for expr in step.exprs:
            expr_origin.setdefault((step.sub_question_id, expr), step.step_id)
        for claim in step.claim_keys:
            claim_origin.setdefault((step.sub_question_id, claim), step.step_id)
        for var in step.variables:
            var_origin.setdefault((step.sub_question_id, var), step.step_id)

    for step in sorted(steps, key=lambda n: n.step_id):
        seen_sources: set[int] = set()

        for expr in step.exprs:
            src = expr_origin.get((step.sub_question_id, expr))
            if src is not None and src < step.step_id and src not in seen_sources:
                edges.append((src, step.step_id, VIRTUAL_EDGE, "expr_ref"))
                evidences.append(EdgeEvidence(src, step.step_id, VIRTUAL_EDGE, "expr_ref", f"expr={expr}"))
                seen_sources.add(src)
                continue
            for (subq, prev_expr), prev_src in expr_origin.items():
                if not _same_sub_question(subq, step.sub_question_id):
                    continue
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
            src = claim_origin.get((step.sub_question_id, claim))
            if src is not None and src < step.step_id and src not in seen_sources:
                edges.append((src, step.step_id, VIRTUAL_EDGE, "claim_ref"))
                evidences.append(EdgeEvidence(src, step.step_id, VIRTUAL_EDGE, "claim_ref", f"claim={claim}"))
                seen_sources.add(src)

        for var in step.variables:
            src = var_origin.get((step.sub_question_id, var))
            if src is not None and src < step.step_id and src not in seen_sources:
                edges.append((src, step.step_id, VIRTUAL_EDGE, "var_ref"))
                evidences.append(EdgeEvidence(src, step.step_id, VIRTUAL_EDGE, "var_ref", f"var={var}"))
                seen_sources.add(src)

        prev = by_id.get(step.step_id - 1)
        if (
            not seen_sources
            and step.step_id > 0
            and prev is not None
            and _same_sub_question(prev.sub_question_id, step.sub_question_id)
            and step.step_type in (StepType.DERIVATION, StepType.CONCLUSION)
            and _should_add_implicit_block_edge(prev, step)
        ):
            edges.append((prev.step_id, step.step_id, DOUBLE_BARRIER_EDGE, "implicit_block"))
            evidences.append(
                EdgeEvidence(prev.step_id, step.step_id, DOUBLE_BARRIER_EDGE, "implicit_block", "fallback")
            )

    return edges, evidences


def _should_add_implicit_block_edge(prev: ParsedStep, step: ParsedStep) -> bool:
    """Gate implicit fallback edges to avoid dominating the edge mix."""
    if not _dag_barrier_strict_enabled():
        return True

    # In strict mode, only add implicit fallback when the current step lacks
    # explicit structural evidence and there is at least weak textual carry-over.
    if step.exprs or step.claim_keys:
        return False
    if prev.step_id >= step.step_id:
        return False
    overlap = _overlap_ratio(prev.normalized_text, step.normalized_text)
    min_overlap = float(os.environ.get("TOPO_DAG_BARRIER_MIN_OVERLAP", "0.2") or 0.2)
    has_prev_signal = bool(prev.exprs or prev.claim_keys or prev.variables)
    return has_prev_signal and overlap >= min_overlap


def build_dependency_edges_by_llm(
    nodes: List[ParsedStep],
    llm_client: Any = None,
) -> Tuple[List[Tuple[int, int, str, str]], List[EdgeEvidence]]:
    """Hybrid edge builder: rule bootstrap + LLM semantic refinement.

    Rule-derived expression/claim/variable edges remain the auditable base.
    A local pretrained LLM can add additional forward-only semantic edges for
    implicit dependencies that do not surface as exact symbolic reuse.  Invalid
    or unavailable LLM outputs never block DAG construction; they only trigger
    a warning and return the rule-based graph.
    """
    rule_edges, rule_evidences = build_dependency_edges_by_rules(nodes)
    # LLM refinement is deliberately opt-in.  It belongs in offline
    # preprocessing where refined DAGs can be cached, not in per-completion
    # GRPO reward calls.
    if not _env_bool("TOPO_DAG_LLM_REFINE", False):
        return rule_edges, rule_evidences
    if len(nodes) < 2:
        return rule_edges, rule_evidences
    max_steps = _env_int("TOPO_DAG_LLM_MAX_STEPS", 16)
    if len(nodes) > max_steps:
        logger.warning(
            "LLM DAG refinement skipped: %d steps exceeds TOPO_DAG_LLM_MAX_STEPS=%d; using rule edges.",
            len(nodes),
            max_steps,
        )
        return rule_edges, rule_evidences

    client = llm_client or _get_local_llm_client()
    if client is None:
        logger.warning("LLM DAG refinement unavailable; using rule-based dependency edges.")
        return rule_edges, rule_evidences

    try:
        llm_edges, llm_evidences = _infer_dependency_edges_with_llm(nodes, client)
    except Exception as exc:
        logger.warning("LLM DAG refinement failed (%s); using rule-based dependency edges.", exc)
        return rule_edges, rule_evidences

    seen = {(s, t, dep) for s, t, _etype, dep in rule_edges}
    merged_edges = list(rule_edges)
    merged_evidences = list(rule_evidences)
    for edge, ev in zip(llm_edges, llm_evidences):
        key = (edge[0], edge[1], edge[3])
        if key in seen:
            continue
        seen.add(key)
        merged_edges.append(edge)
        merged_evidences.append(ev)
    return merged_edges, merged_evidences


class _LocalHFDependencyClient:
    def __init__(self, model_path: str, device: str, max_new_tokens: int) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        kwargs: Dict[str, Any] = {"trust_remote_code": True}
        if device == "auto":
            kwargs["device_map"] = "auto"
        else:
            kwargs["device_map"] = {"": device}
        kwargs["torch_dtype"] = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
        self.max_new_tokens = max_new_tokens

    def generate(self, prompt: str) -> str:
        messages = [
            {"role": "system", "content": "You extract dependency edges between reasoning steps. Return JSON only."},
            {"role": "user", "content": prompt},
        ]
        if hasattr(self.tokenizer, "apply_chat_template"):
            text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            text = f"{messages[0]['content']}\n\n{messages[1]['content']}\nJSON:"
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        out = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
        )
        decoded = self.tokenizer.decode(out[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
        return decoded.strip()


def _get_local_llm_client() -> Any:
    global _LOCAL_LLM_CLIENT, _LOCAL_LLM_LOAD_ATTEMPTED
    if _LOCAL_LLM_CLIENT is not None:
        return _LOCAL_LLM_CLIENT
    if _LOCAL_LLM_LOAD_ATTEMPTED:
        return None
    _LOCAL_LLM_LOAD_ATTEMPTED = True

    model_path = os.environ.get(
        "TOPO_DAG_LLM_MODEL",
        "${HF_MODELS_DIR:-./models}/Qwen/Qwen2.5-Math-1.5B-Instruct",
    )
    if not model_path or not Path(model_path).exists():
        logger.warning("LLM DAG refinement model not found at %s; using rule-based fallback.", model_path)
        return None
    try:
        _LOCAL_LLM_CLIENT = _LocalHFDependencyClient(
            model_path=model_path,
            device=os.environ.get("TOPO_DAG_LLM_DEVICE", "auto"),
            max_new_tokens=_env_int("TOPO_DAG_LLM_MAX_NEW_TOKENS", 512),
        )
    except Exception as exc:
        logger.warning("Failed to load local LLM DAG refiner (%s); using rule-based fallback.", exc)
        _LOCAL_LLM_CLIENT = None
    return _LOCAL_LLM_CLIENT


def _dependency_prompt(nodes: List[ParsedStep]) -> str:
    rows = []
    for n in nodes:
        rows.append(
            {
                "id": n.step_id,
                "text": n.raw_text,
                "type": n.step_type.value,
                "exprs": n.exprs[:8],
                "claims": n.claims[:4],
                "variables": n.variables[:8],
            }
        )
    return (
        "Task: identify implicit semantic dependencies between reasoning steps that are NOT already captured "
        "by exact symbolic reuse. Add an edge i -> j only if step j semantically depends on an intermediate "
        "result, latent claim, sub-goal, or assumption introduced in step i.\n"
        "\n"
        "Output STRICT JSON only. No prose, no markdown, no comments. The output MUST be a JSON array.\n"
        "Each item MUST be an object with exactly these keys:\n"
        "  - source: integer step id\n"
        "  - target: integer step id\n"
        "  - dep_type: one of [\"llm_semantic\", \"llm_subgoal\"]\n"
        "  - confidence: float in [0,1] reflecting how confident you are in this implicit edge\n"
        "  - evidence: a short English phrase (<= 24 words) explaining the semantic link\n"
        "\n"
        "Hard constraints:\n"
        "  - source and target must both be valid step ids from the input\n"
        "  - source < target (forward-only, no cycles)\n"
        "  - no self edges\n"
        "  - skip dependencies that are already obvious from shared expressions, variables, or claim keys; "
        "    only output IMPLICIT semantic dependencies\n"
        "  - if no implicit dependency exists, output []\n"
        "\n"
        "Worked example.\n"
        "Input steps (parsed):\n"
        "[\n"
        "  {\"id\":0,\"text\":\"Let n be a positive integer.\",\"type\":\"definition\",\"exprs\":[],\"claims\":[],\"variables\":[\"n\"]},\n"
        "  {\"id\":1,\"text\":\"Assume n is even.\",\"type\":\"definition\",\"exprs\":[],\"claims\":[\"n is even\"],\"variables\":[\"n\"]},\n"
        "  {\"id\":2,\"text\":\"Then n^2 is even by parity.\",\"type\":\"derivation\",\"exprs\":[],\"claims\":[\"n^2 is even\"],\"variables\":[\"n\"]},\n"
        "  {\"id\":3,\"text\":\"Compute n^2 - n = n(n-1).\",\"type\":\"computation\",\"exprs\":[\"n^2-n=n(n-1)\"],\"claims\":[],\"variables\":[\"n\"]},\n"
        "  {\"id\":4,\"text\":\"So n(n-1) is even.\",\"type\":\"conclusion\",\"exprs\":[],\"claims\":[\"n(n-1) is even\"],\"variables\":[\"n\"]}\n"
        "]\n"
        "Expected output (only IMPLICIT dependencies, e.g. step 4 reuses the parity argument from step 1):\n"
        "[\n"
        "  {\"source\":1,\"target\":4,\"dep_type\":\"llm_semantic\",\"confidence\":0.85,\"evidence\":\"step 4 reuses the parity assumption introduced in step 1\"}\n"
        "]\n"
        "\n"
        f"Now process the following steps:\nSteps:\n{json.dumps(rows, ensure_ascii=False, indent=2)}"
    )


def _extract_json_array(text: str) -> list[Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text).strip()
        text = re.sub(r"```$", "", text).strip()
    start = text.find("[")
    end = text.rfind("]")
    if start < 0 or end < start:
        raise ValueError("LLM output does not contain a JSON array")
    return json.loads(text[start:end + 1])


def _infer_dependency_edges_with_llm(
    nodes: List[ParsedStep],
    llm_client: Any,
) -> Tuple[List[Tuple[int, int, str, str]], List[EdgeEvidence]]:
    prompt = _dependency_prompt(nodes)
    raw = llm_client.generate(prompt) if hasattr(llm_client, "generate") else llm_client(prompt)
    rows = _extract_json_array(str(raw))
    valid_ids = {n.step_id for n in nodes}
    edges: List[Tuple[int, int, str, str]] = []
    evidences: List[EdgeEvidence] = []
    seen: set[tuple[int, int]] = set()
    allowed_dep_types = {"llm_semantic", "llm_subgoal"}
    for item in rows:
        if not isinstance(item, dict):
            continue
        try:
            src = int(item.get("source"))
            tgt = int(item.get("target"))
        except (TypeError, ValueError):
            continue
        if src not in valid_ids or tgt not in valid_ids or src >= tgt or (src, tgt) in seen:
            continue
        dep_type = str(item.get("dep_type", "llm_semantic"))
        if dep_type not in allowed_dep_types:
            dep_type = "llm_semantic"
        try:
            confidence = float(item.get("confidence", 0.5))
        except (TypeError, ValueError):
            confidence = 0.5
        confidence = max(0.0, min(1.0, confidence))
        seen.add((src, tgt))
        evidence = str(item.get("evidence", "llm semantic dependency"))[:160]
        if confidence > 0:
            evidence = f"[conf={confidence:.2f}] {evidence}"
        edges.append((src, tgt, VIRTUAL_EDGE, dep_type))
        evidences.append(EdgeEvidence(src, tgt, VIRTUAL_EDGE, dep_type, evidence))
    return edges, evidences


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
    if _dag_seq_when_no_dep_only():
        # Stricter: require the source to also have no structural in-edge,
        # so chains form only across nodes with no symbolic carry.
        for u, v, data in dag.graph.in_edges(src_id, data=True):
            _ = (u, v)
            et = data.get("edge_type", "")
            if dag.is_virtual_edge(et) or et == DOUBLE_BARRIER_EDGE:
                return False
    return True


def _add_sequential_weak_edges(
    dag: ReasoningDAG,
    parsed_steps: List[ParsedStep],
) -> int:
    ids = sorted(dag.nodes)
    sid_to_subq = {s.step_id: s.sub_question_id for s in parsed_steps}
    added = 0
    for a, b in zip(ids, ids[1:]):
        if not _same_sub_question(sid_to_subq.get(a), sid_to_subq.get(b)):
            continue
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
    if re.search(r"错误|不成立|矛盾|有误|invalid|wrong|contradiction|impossible|no solution", text, re.IGNORECASE):
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

    dep_edges, evidences = build_dependency_edges_by_llm(parsed_steps)
    for src_id, tgt_id, edge_type, dep_type in dep_edges:
        if edge_type == VIRTUAL_EDGE:
            dag.add_dependency_edge(src_id, tgt_id, dep_type)
        else:
            dag.add_implicit_barrier_edge(src_id, tgt_id)

    seq_edges_added = 0
    if _ENABLE_SEQ_WEAK_EDGE:
        seq_edges_added = _add_sequential_weak_edges(dag, parsed_steps)

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
        "sub_questions": sorted({s.sub_question_id for s in parsed_steps if s.sub_question_id is not None}),
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


def _try_build_from_native_dag(native_dag: Dict[str, Any], problem_id: str) -> Optional[ReasoningDAG]:
    """Build a ReasoningDAG from a native DAG specification.

    Returns None if the native_dag is malformed or empty, allowing fallback
    to rule-based construction. This preserves backward compatibility:
    records without native dag use the original pipeline unchanged.
    """
    nodes_raw = native_dag.get("nodes")
    edges_raw = native_dag.get("edges")
    if not nodes_raw or not isinstance(nodes_raw, list):
        return None

    dag = ReasoningDAG(problem_id=problem_id)
    _v2_type_map = {
        "decompose": StepType.DEFINITION,
        "derive": StepType.DERIVATION,
        "check": StepType.COMPUTATION,
        "conclude": StepType.CONCLUSION,
        "definition": StepType.DEFINITION,
        "auxiliary": StepType.AUXILIARY,
    }
    _v2_verdict_map = {
        "correct": LocalVerdict.CORRECT,
        "incorrect": LocalVerdict.INCORRECT,
        "unverifiable": LocalVerdict.UNVERIFIABLE,
    }
    id_to_int: Dict[str, int] = {}
    for idx, n in enumerate(nodes_raw):
        if not isinstance(n, dict):
            continue
        nid = str(n.get("id", f"n{idx}"))
        int_id = idx
        id_to_int[nid] = int_id
        raw_text = str(n.get("text", ""))
        stype = _v2_type_map.get(str(n.get("type", "")), StepType.UNKNOWN)
        verdict = _v2_verdict_map.get(str(n.get("local_verdict", "")), LocalVerdict.UNVERIFIABLE)
        dag.add_node(Node(
            step_id=int_id,
            raw_text=raw_text,
            normalized_text=raw_text,
            exprs=extract_expressions(raw_text),
            claims=extract_claims(raw_text),
            step_type=stype,
            local_verdict=verdict,
            sub_question_id=n.get("sub_question_id"),
        ))

    if edges_raw and isinstance(edges_raw, list):
        for e in edges_raw:
            if not isinstance(e, dict):
                continue
            src_str = str(e.get("from", ""))
            tgt_str = str(e.get("to", ""))
            src_int = id_to_int.get(src_str)
            tgt_int = id_to_int.get(tgt_str)
            if src_int is None or tgt_int is None:
                continue
            rel = str(e.get("rel", "depends_on"))
            if rel == "contradicts":
                dag.add_implicit_barrier_edge(src_int, tgt_int)
            else:
                dag.add_dependency_edge(src_int, tgt_int, rel)

    if dag.num_nodes == 0:
        return None
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
    native_used = 0
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

            rid = record.get("record_id", f"record_{line_no}")

            # Native DAG first, fallback to rule-based construction.
            # Keep `v2_dag` as backward-compatible alias.
            native_dag_raw = record.get("native_dag") or record.get("v2_dag")
            dag = None
            if isinstance(native_dag_raw, dict) and native_dag_raw:
                dag = _try_build_from_native_dag(native_dag_raw, problem_id=rid)
                if dag is not None:
                    native_used += 1

            if dag is None:
                answer = record.get("standard_answer", "")
                dag = build_dag_from_answer(answer, problem_id=rid)

            out_file = out_dir / f"{rid}.json"
            out_file.write_text(dag.to_json(), encoding="utf-8")
            count += 1

    logger.info("Built %d DAGs -> %s (native=%d, rule_fallback=%d)",
                count, out_dir, native_used, count - native_used)


if __name__ == "__main__":
    main()

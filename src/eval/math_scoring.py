from __future__ import annotations

import re

from math_verify import ExprExtractionConfig, LatexExtractionConfig, parse, verify


_BOXED_START_RE = re.compile(r"\\boxed\{")
_FINAL_ANSWER_RE = re.compile(
    r"(?is)(?:"
    r"\bfinal\s+answer\b\s*(?:is\s*)?[:=]?"
    r"|\b(?:the\s+)?answer\s+(?:is|equals)\s*[:=]?"
    r"|\banswer\s*[:=]\s*"
    r")"
)
_MAX_EXPLICIT_FINAL_CHARS = 256
_MAX_EXPLICIT_FINAL_WORDS = 64


def extract_last_boxed(text: str) -> str | None:
    answers: list[str] = []
    for match in _BOXED_START_RE.finditer(str(text)):
        start = match.end()
        depth = 1
        for index in range(start, len(text)):
            char = text[index]
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    answers.append(text[start:index].strip())
                    break
    return next((answer for answer in reversed(answers) if answer), None)


def extract_explicit_final_answer(text: str) -> str | None:
    """Return a terminal, explicitly marked answer region.

    Long continuations usually indicate that the marker occurred inside an
    unfinished reasoning or self-check block rather than in a submitted answer.
    """
    matches = list(_FINAL_ANSWER_RE.finditer(str(text)))
    if not matches:
        return None
    candidate = str(text)[matches[-1].end() :].strip()
    if not candidate:
        return None
    if len(candidate) > _MAX_EXPLICIT_FINAL_CHARS:
        return None
    if len(candidate.split()) > _MAX_EXPLICIT_FINAL_WORDS:
        return None
    return candidate


def parse_answer_candidate(value: str) -> list:
    candidate = str(value).strip()
    if not candidate:
        return []
    configs = [LatexExtractionConfig(boxed_match_priority=0), ExprExtractionConfig()]
    parsed = parse(r"\boxed{" + candidate + "}", extraction_config=configs)
    return parsed or parse(candidate, extraction_config=configs)


def verify_answer_equivalence(prediction: str, ground_truth: str) -> bool:
    parsed_gt = parse_answer_candidate(ground_truth)
    parsed_pred = parse_answer_candidate(prediction)
    if parsed_pred and parsed_gt:
        return verify(parsed_gt, parsed_pred)
    pred_clean = str(prediction).strip().rstrip(".").strip()
    gt_clean = str(ground_truth).strip().rstrip(".").strip()
    return pred_clean == gt_clean


def verify_math_response(response: str, ground_truth: str) -> bool:
    parsed_gt = parse_answer_candidate(ground_truth)
    if not parsed_gt:
        return False
    boxed = extract_last_boxed(str(response))
    if boxed is not None:
        candidate = boxed
    else:
        candidate = extract_explicit_final_answer(str(response))
    if candidate is None:
        return False
    parsed_pred = parse_answer_candidate(candidate)
    return bool(parsed_pred and verify(parsed_gt, parsed_pred))

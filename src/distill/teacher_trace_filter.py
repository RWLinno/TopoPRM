from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.data.generate_distill_data import score_trace_quality


@dataclass
class TraceFilterConfig:
    min_quality: float = 0.6
    max_chars: int = 1600
    target_chars: int = 900
    length_weight: float = 0.15


def _composite_score(text: str, quality: float, cfg: TraceFilterConfig) -> float:
    """Quality-first score with soft length penalty for compression."""
    n_chars = len(text)
    overflow = max(0.0, (n_chars - cfg.target_chars) / max(cfg.target_chars, 1))
    return float(quality) - float(cfg.length_weight) * float(overflow)


def _thinking_mode(n_chars: int, target_chars: int) -> str:
    if n_chars <= int(target_chars * 0.8):
        return 'fast'
    if n_chars >= int(target_chars * 1.2):
        return 'slow'
    return 'adaptive'


def process_aware_trace_filter(responses: list[str], config: TraceFilterConfig | None = None) -> list[tuple[str, float]]:
    """Filter traces by quality and compactness to favor compressible reasoning."""
    cfg = config or TraceFilterConfig()
    kept: list[tuple[str, float]] = []
    for r in responses:
        q = float(score_trace_quality(r))
        if q < cfg.min_quality or len(r) > cfg.max_chars:
            continue
        s = _composite_score(r, q, cfg)
        kept.append((r, s))
    kept.sort(key=lambda x: x[1], reverse=True)
    return kept


def _extract_assistant_text(row: dict[str, Any]) -> str:
    for k in ('response', 'prediction', 'output', 'text'):
        if isinstance(row.get(k), str):
            return row[k]
    msgs = row.get('messages')
    if isinstance(msgs, list):
        for m in reversed(msgs):
            if isinstance(m, dict) and m.get('role') == 'assistant':
                return str(m.get('content', ''))
    return ''


def main() -> None:
    parser = argparse.ArgumentParser(description='Filter teacher traces for distillation')
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--min_quality', type=float, default=0.6)
    parser.add_argument('--max_chars', type=int, default=1600)
    parser.add_argument('--target_chars', type=int, default=900)
    parser.add_argument('--length_weight', type=float, default=0.15)
    parser.add_argument('--keep_top_k', type=int, default=0)
    args = parser.parse_args()

    cfg = TraceFilterConfig(
        min_quality=float(args.min_quality),
        max_chars=int(args.max_chars),
        target_chars=int(args.target_chars),
        length_weight=float(args.length_weight),
    )

    rows: list[dict[str, Any]] = []
    with args.input.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    scored: list[tuple[float, dict[str, Any]]] = []
    for r in rows:
        txt = _extract_assistant_text(r)
        if not txt:
            continue
        q = float(score_trace_quality(txt))
        if q < cfg.min_quality or len(txt) > cfg.max_chars:
            continue

        comp = _composite_score(txt, q, cfg)
        r['distill_quality_score'] = round(q, 4)
        r['distill_composite_score'] = round(comp, 4)
        r['distill_num_chars'] = int(len(txt))
        r['thinking_mode'] = _thinking_mode(len(txt), cfg.target_chars)
        scored.append((comp, r))

    scored.sort(key=lambda x: x[0], reverse=True)
    if args.keep_top_k and args.keep_top_k > 0:
        scored = scored[: args.keep_top_k]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('w', encoding='utf-8') as f:
        for _, r in scored:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')

    print(f'[teacher_trace_filter] kept {len(scored)} / {len(rows)} -> {args.output}')


if __name__ == '__main__':
    main()

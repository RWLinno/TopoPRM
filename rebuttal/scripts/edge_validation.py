#!/usr/bin/env python3
"""LLM-based edge validation for the TopoPRM DAG extractor.

Independent-judge protocol (answers HxUk W1, B5w7 W1, TsKG W1):

  1. Sample N traces from data/grpo_ready/train_public.jsonl, stratified by
     source (gsm8k / math) and trace length (num steps).
  2. For each trace, re-run the extractor (src.data.build_dag) to obtain the
     candidate support edges and the segmented steps.
  3. Ask a strong LLM, *blind to the extractor edges*, to judge for every
     ordered step pair (i<j) whether step i provides necessary support for
     step j. The union of LLM "yes" pairs is the reference edge set A*.
  4. Compare extractor edges A_E against A* -> precision / recall / F1 /
     per-dep-type reliability, plus a false-positive taxonomy.

Two phases so the expensive LLM calls are decoupled from sampling:

    python edge_validation.py sample   --n 120 --out <pack.jsonl>
    python edge_validation.py annotate --pack <pack.jsonl> --out <ann.jsonl>
    python edge_validation.py score    --pack <pack.jsonl> --ann <ann.jsonl> --out <results.json>

The annotate phase is model-agnostic (any OpenAI-compatible endpoint via
--base-url / --model, key from OPENAI_API_KEY or --api-key).
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.data.build_dag import build_dag_from_answer, extract_steps_from_answer  # noqa: E402

DATA = REPO / "data" / "grpo_ready" / "train_public.jsonl"


def _steps_and_edges(answer: str) -> tuple[list[str], list[dict[str, Any]]]:
    steps = extract_steps_from_answer(answer)
    texts = [s.get("normalized_text") or s.get("raw_text", "") for s in steps]
    dag = build_dag_from_answer(answer)
    edges = [
        {
            "source": e.source,
            "target": e.target,
            "edge_type": e.edge_type,
            "dep_type": getattr(e, "dep_type", None),
        }
        for e in dag.edges
    ]
    return texts, edges


def cmd_sample(args: argparse.Namespace) -> None:
    rng = random.Random(args.seed)
    rows = [json.loads(l) for l in DATA.open() if l.strip()]
    # bucket by (source, length-band)
    buckets: dict[tuple, list] = defaultdict(list)
    prepared = []
    for r in rows:
        texts, edges = _steps_and_edges(r["standard_answer"])
        n = len(texts)
        if n < 2 or n > 12:  # need >=2 steps to have any edge; cap for annotation cost
            continue
        band = "short" if n <= 3 else ("mid" if n <= 6 else "long")
        rec = {
            "record_id": r["record_id"],
            "source": r["source"],
            "question": r["question"],
            "final_answer": r["final_answer"],
            "steps": texts,
            "extractor_edges": edges,
            "n_steps": n,
            "band": band,
        }
        buckets[(r["source"], band)].append(rec)
    # even allocation across buckets
    keys = sorted(buckets)
    per = max(1, args.n // len(keys))
    for k in keys:
        rng.shuffle(buckets[k])
        prepared.extend(buckets[k][:per])
    rng.shuffle(prepared)
    prepared = prepared[: args.n]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for rec in prepared:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    dist = Counter((r["source"], r["band"]) for r in prepared)
    n_edges = sum(len(r["extractor_edges"]) for r in prepared)
    print(f"[sample] wrote {len(prepared)} traces, {n_edges} extractor edges -> {out}")
    print(f"[sample] strata: {dict(dist)}")


_JUDGE_SYS = (
    "You are a meticulous mathematics grader. You are given a math problem and a "
    "solution that has been split into numbered steps. Your job is to identify the "
    "SUPPORT DEPENDENCIES between steps: step i supports step j (i<j) if the result, "
    "quantity, or established fact in step i is NECESSARY to derive or justify step j. "
    "Ignore mere textual similarity or shared words that are not logically used. "
    "Only output dependencies you are confident a human grader would agree with."
)


def _judge_prompt(rec: dict[str, Any]) -> str:
    lines = [f"Problem: {rec['question']}", "", "Steps:"]
    for i, s in enumerate(rec["steps"]):
        lines.append(f"[{i}] {s}")
    lines += [
        "",
        "For every ordered pair (i, j) with i < j where step i provides NECESSARY "
        "support for step j, identify the pair. You may reason briefly, but you "
        "MUST end your reply with a line of the exact form:",
        "ANSWER: [[i,j], ...]",
        "where the value is a JSON array of [i,j] integer pairs (use [] if there "
        "are no dependencies). The ANSWER line must be the last line.",
    ]
    return "\n".join(lines)


def _parse_pairs(text: str, n: int) -> list[list[int]]:
    # Prefer content after an explicit answer marker (handles thinking models).
    for marker in ("ANSWER:", "Answer:", "</think>", "Final answer:", "FINAL:"):
        if marker in text:
            text = text.split(marker)[-1]
            break
    # Find ALL bracketed arrays of pairs and take the last parseable one that
    # looks like a list of [i, j] pairs (thinking models restate the array).
    candidates = re.findall(r"\[\s*(?:\[\s*\d+\s*,\s*\d+\s*\]\s*,?\s*)*\]", text, re.DOTALL)
    arr = None
    for cand in reversed(candidates):
        try:
            parsed = json.loads(cand)
        except Exception:
            continue
        if isinstance(parsed, list) and (not parsed or isinstance(parsed[0], list)):
            arr = parsed
            break
    if arr is None:
        return []
    out = []
    seen = set()
    for p in arr:
        if isinstance(p, (list, tuple)) and len(p) == 2:
            try:
                i, j = int(p[0]), int(p[1])
            except (ValueError, TypeError):
                continue
            if 0 <= i < j < n and (i, j) not in seen:
                seen.add((i, j))
                out.append([i, j])
    return out


def cmd_annotate_local(args: argparse.Namespace) -> None:
    """Independent judge via a local HF transformers model (no external API)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    recs = [json.loads(l) for l in Path(args.pack).open() if l.strip()]
    out = Path(args.out)
    done: set[str] = set()
    if out.exists() and args.resume:
        for l in out.open():
            if l.strip():
                done.add(json.loads(l)["record_id"])
    todo = [r for r in recs if r["record_id"] not in done]
    if not todo:
        print("[annotate-local] nothing to do")
        return

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )
    model.eval()
    print(f"[annotate-local] loaded {args.model}; {len(todo)} traces to judge", flush=True)

    def _gen(rec: dict[str, Any]) -> str:
        msgs = [
            {"role": "system", "content": _JUDGE_SYS},
            {"role": "user", "content": _judge_prompt(rec)},
        ]
        ct_kwargs = dict(tokenize=False, add_generation_prompt=True)
        if "enable_thinking" in _chat_kwargs(tok):
            ct_kwargs["enable_thinking"] = False  # Qwen3: disable thinking for JSON output
        text = tok.apply_chat_template(msgs, **ct_kwargs)
        inputs = tok(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            gen = model.generate(
                **inputs, max_new_tokens=args.max_tokens, do_sample=False,
                pad_token_id=tok.pad_token_id or tok.eos_token_id,
            )
        return tok.decode(gen[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    with out.open("a" if args.resume else "w") as f:
        for k, rec in enumerate(todo):
            try:
                content = _gen(rec)
                pairs = _parse_pairs(content, rec["n_steps"])
                err = None
            except Exception as e:  # noqa: BLE001
                content, pairs, err = "", [], str(e)[:160]
            f.write(
                json.dumps(
                    {
                        "record_id": rec["record_id"],
                        "n_steps": rec["n_steps"],
                        "llm_edges": pairs,
                        "raw": content[-400:],
                        "error": err,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            f.flush()
            print(f"[annotate-local {k+1}/{len(todo)}] {rec['record_id']} "
                  f"{'ERR' if err else 'ok'} edges={len(pairs)}", flush=True)
    print(f"[annotate-local] wrote {len(todo)} annotations -> {out}")


def _chat_kwargs(tok) -> set:
    import inspect
    try:
        return set(inspect.signature(tok.apply_chat_template).parameters)
    except Exception:
        return set()


def cmd_annotate(args: argparse.Namespace) -> None:
    if getattr(args, "local", False):
        return cmd_annotate_local(args)
    from openai import OpenAI

    key = args.api_key or os.environ.get("OPENAI_API_KEY", "")
    client = OpenAI(base_url=args.base_url, api_key=key, timeout=args.timeout)
    recs = [json.loads(l) for l in Path(args.pack).open() if l.strip()]
    out = Path(args.out)
    done: dict[str, Any] = {}
    if out.exists() and args.resume:
        for l in out.open():
            if l.strip():
                d = json.loads(l)
                done[d["record_id"]] = d
    with out.open("a" if args.resume else "w") as f:
        for k, rec in enumerate(recs):
            if rec["record_id"] in done:
                continue
            prompt = _judge_prompt(rec)
            content, err = "", None
            for attempt in range(args.retries):
                try:
                    r = client.chat.completions.create(
                        model=args.model,
                        messages=[
                            {"role": "system", "content": _JUDGE_SYS},
                            {"role": "user", "content": prompt},
                        ],
                        max_tokens=512,
                        temperature=0.0,
                    )
                    content = r.choices[0].message.content or ""
                    err = None
                    break
                except Exception as e:  # noqa: BLE001
                    err = str(e)[:160]
                    time.sleep(2 * (attempt + 1))
            pairs = _parse_pairs(content, rec["n_steps"]) if not err else []
            rowout = {
                "record_id": rec["record_id"],
                "n_steps": rec["n_steps"],
                "llm_edges": pairs,
                "raw": content,
                "error": err,
            }
            f.write(json.dumps(rowout, ensure_ascii=False) + "\n")
            f.flush()
            tag = "ERR" if err else "ok"
            print(f"[annotate {k+1}/{len(recs)}] {rec['record_id']} {tag} edges={len(pairs)}", flush=True)


def cmd_score(args: argparse.Namespace) -> None:
    packs = {r["record_id"]: r for r in (json.loads(l) for l in Path(args.pack).open() if l.strip())}
    anns = {r["record_id"]: r for r in (json.loads(l) for l in Path(args.ann).open() if l.strip())}

    tp = fp = fn = 0
    by_type_tp: Counter = Counter()
    by_type_fp: Counter = Counter()
    fp_examples: list[dict[str, Any]] = []
    per_trace = []
    n_used = 0
    for rid, pack in packs.items():
        ann = anns.get(rid)
        if ann is None or ann.get("error"):
            continue
        n_used += 1
        gold = {tuple(p) for p in ann["llm_edges"]}
        ext = {(e["source"], e["target"]): e for e in pack["extractor_edges"]}
        ext_set = set(ext)
        t_tp = len(ext_set & gold)
        t_fp = len(ext_set - gold)
        t_fn = len(gold - ext_set)
        tp += t_tp
        fp += t_fp
        fn += t_fn
        for pr in ext_set & gold:
            by_type_tp[ext[pr].get("dep_type") or ext[pr]["edge_type"]] += 1
        for pr in ext_set - gold:
            dt = ext[pr].get("dep_type") or ext[pr]["edge_type"]
            by_type_fp[dt] += 1
            if len(fp_examples) < 12:
                fp_examples.append(
                    {
                        "record_id": rid,
                        "edge": list(pr),
                        "dep_type": dt,
                        "src": pack["steps"][pr[0]],
                        "tgt": pack["steps"][pr[1]],
                    }
                )
        per_trace.append({"record_id": rid, "tp": t_tp, "fp": t_fp, "fn": t_fn})

    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0

    type_reliability = {}
    for dt in set(list(by_type_tp) + list(by_type_fp)):
        d_tp, d_fp = by_type_tp[dt], by_type_fp[dt]
        type_reliability[dt] = {
            "precision": round(d_tp / (d_tp + d_fp), 4) if (d_tp + d_fp) else 0.0,
            "tp": d_tp,
            "fp": d_fp,
        }

    result = {
        "n_traces_scored": n_used,
        "n_traces_total": len(packs),
        "edges": {"tp": tp, "fp": fp, "fn": fn},
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "per_dep_type": type_reliability,
        "false_positive_examples": fp_examples,
        "judge_model": args.model,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != "false_positive_examples"}, indent=2))
    print(f"[score] -> {args.out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("sample")
    s.add_argument("--n", type=int, default=120)
    s.add_argument("--seed", type=int, default=42)
    s.add_argument("--out", default="rebuttal/outputs/edge_validation_pack.jsonl")
    s.set_defaults(func=cmd_sample)

    a = sub.add_parser("annotate")
    a.add_argument("--pack", default="rebuttal/outputs/edge_validation_pack.jsonl")
    a.add_argument("--out", default="rebuttal/outputs/edge_validation_annotations.jsonl")
    a.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    a.add_argument("--model", default=os.environ.get("EDGE_JUDGE_MODEL", "gpt-4o"))
    a.add_argument("--api-key", default=None)
    a.add_argument("--timeout", type=float, default=60.0)
    a.add_argument("--retries", type=int, default=3)
    a.add_argument("--resume", action="store_true", default=True)
    a.add_argument("--local", action="store_true", help="use local vLLM judge instead of external API")
    a.add_argument("--tp", type=int, default=2, help="tensor parallel size for local vLLM")
    a.add_argument("--max-model-len", type=int, default=8192)
    a.add_argument("--max-tokens", type=int, default=512)
    a.set_defaults(func=cmd_annotate)

    c = sub.add_parser("score")
    c.add_argument("--pack", default="rebuttal/outputs/edge_validation_pack.jsonl")
    c.add_argument("--ann", default="rebuttal/outputs/edge_validation_annotations.jsonl")
    c.add_argument("--out", default="rebuttal/outputs/edge_validation_results.json")
    c.add_argument("--model", default=os.environ.get("EDGE_JUDGE_MODEL", "gpt-4o"))
    c.set_defaults(func=cmd_score)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

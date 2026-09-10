#!/usr/bin/env python3
"""Prepare AIME2026 + FrontierMath benchmark datasets.

Outputs:
  - data/benchmarks/AIME2026/train.jsonl
  - data/benchmarks/FrontierMath/train.jsonl
"""

from __future__ import annotations

import argparse
import io
import json
import re
import urllib.request
import zipfile
from collections import OrderedDict
from pathlib import Path
from typing import Any

from datasets import load_dataset


AIME2026_CANDIDATES = [
    {"path": "MathArena/aime_2026"},
    {"path": "MathArena/aime_2026_I"},
    {"path": "math-ai/aime26"},
]

FRONTIER_SAMPLE_ZIP = "https://epoch.ai/files/sample_question_transcripts.zip"


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _norm_problem(row: dict[str, Any]) -> str:
    for k in ("problem", "Problem", "question", "Question", "prompt"):
        v = row.get(k)
        if v is not None and str(v).strip():
            return str(v).strip()
    return ""


def _norm_answer(row: dict[str, Any]) -> str:
    for k in ("answer", "Answer", "final_answer", "ground_truth"):
        v = row.get(k)
        if v is not None and str(v).strip():
            return str(v).strip()
    return ""


def prepare_aime2026(out_path: Path) -> dict[str, Any]:
    errors: list[str] = []
    for cand in AIME2026_CANDIDATES:
        try:
            ds = load_dataset(cand["path"], split="train")
            rows = []
            for i, item in enumerate(ds):
                problem = _norm_problem(item)
                answer = _norm_answer(item)
                if not problem:
                    continue
                rows.append(
                    {
                        "id": item.get("id", item.get("problem_idx", i)),
                        "problem": problem,
                        "answer": answer,
                        "source": cand["path"],
                    }
                )
            if rows:
                _write_jsonl(out_path, rows)
                return {"ok": True, "source": cand["path"], "num_rows": len(rows)}
        except Exception as e:  # noqa: BLE001
            errors.append(f"{cand['path']}: {e}")
    return {"ok": False, "errors": errors}


def _extract_frontier_problem(content: str) -> tuple[str, str]:
    prompt_re = re.compile(
        r"Here is the mathematical problem you need to solve:\s*(.*?)\s*The expected return type of final_answer is the following:\s*(.+?)\s*$",
        re.DOTALL,
    )
    m = prompt_re.search(content)
    if not m:
        return "", ""
    problem = m.group(1).strip()
    expected_type = m.group(2).strip()
    return problem, expected_type


def prepare_frontiermath(out_path: Path) -> dict[str, Any]:
    req = urllib.request.Request(
        FRONTIER_SAMPLE_ZIP,
        headers={"User-Agent": "Mozilla/5.0"},
    )
    rows_by_problem: "OrderedDict[str, dict[str, Any]]" = OrderedDict()
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:  # noqa: S310
            payload = resp.read()
    except Exception as e:
        return {"ok": False, "errors": [f"Download failed: {e}"]}

    with zipfile.ZipFile(io.BytesIO(payload)) as zf:
        jsonl_files = [n for n in zf.namelist() if n.endswith(".jsonl")]
        for name in jsonl_files:
            run_tag = Path(name).name.replace(".jsonl", "")
            lines = zf.read(name).decode("utf-8", errors="ignore").splitlines()
            for ln in lines:
                try:
                    obj = json.loads(ln)
                except json.JSONDecodeError:
                    continue
                if obj.get("role") != "user":
                    continue
                content = str(obj.get("content", ""))
                problem, expected_type = _extract_frontier_problem(content)
                if not problem:
                    continue
                if problem not in rows_by_problem:
                    rows_by_problem[problem] = {
                        "id": f"frontier_{len(rows_by_problem)}",
                        "problem": problem,
                        "answer": "",
                        "expected_return_type": expected_type,
                        "source": "epoch_frontiermath_sample",
                        "sample_run": run_tag,
                    }

    rows = list(rows_by_problem.values())
    if not rows:
        return {"ok": False, "errors": ["No FrontierMath samples parsed from zip"]}
    _write_jsonl(out_path, rows)
    return {"ok": True, "source": FRONTIER_SAMPLE_ZIP, "num_rows": len(rows)}


def update_manifest_status(bench_root: Path, updates: dict[str, dict[str, Any]]) -> None:
    manifest_path = bench_root / "manifest.json"
    status_path = bench_root / "status.md"
    manifest = {}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.setdefault("downloaded", {})
    manifest.setdefault("tbd", {})
    for name, info in updates.items():
        if info.get("ok"):
            manifest["downloaded"][name] = {
                "source": {"path": info.get("source", "manual")},
                "note": "English",
                "split_sizes": {"train": int(info.get("num_rows", 0))},
            }
            manifest["tbd"].pop(name, None)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Benchmark Download Status",
        "",
        "Generated by scripts/prepare_frontiermath_aime2026.py",
        "",
        "## Downloaded",
    ]
    downloaded = manifest.get("downloaded", {})
    for name, info in downloaded.items():
        lines.append(f"- {name}: source={info.get('source')} splits={info.get('split_sizes')}")
    lines.extend(["", "## TBD"])
    tbd = manifest.get("tbd", {})
    if tbd:
        for name, info in tbd.items():
            lines.append(f"- {name}: note={info.get('note', '')}")
    else:
        lines.append("- (none)")
    status_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench_root", type=Path, default=Path("data/benchmarks"))
    args = parser.parse_args()

    bench_root = args.bench_root
    aime_path = bench_root / "AIME2026" / "train.jsonl"
    frontier_path = bench_root / "FrontierMath" / "train.jsonl"

    print("Preparing AIME 2026...")
    aime_res = prepare_aime2026(aime_path)
    print(f"  AIME2026: {aime_res}")

    print("Preparing FrontierMath...")
    frontier_res = prepare_frontiermath(frontier_path)
    print(f"  FrontierMath: {frontier_res}")

    update_manifest_status(bench_root, {"AIME2026": aime_res, "FrontierMath": frontier_res})

    print(json.dumps({"AIME2026": aime_res, "FrontierMath": frontier_res}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

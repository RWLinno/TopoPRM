#!/usr/bin/env python3
"""Generate ``docs/dag_showcase.html`` from cached audit DAGs and case PNGs.

Reads ``output/dag_audit/dag_audit_dr1_7b_<bench>_cached.jsonl`` for every
benchmark that has both a cached jsonl and at least one rendered PNG case in
``docs/assets/dag_cases/<bench>/``.  For each benchmark, picks up to ``--n``
cases (preferring traces with the most non-implicit edges) and emits a
self-contained HTML page with structured node / edge details.
"""

from __future__ import annotations

import argparse
import html
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LABEL = "dag_audit_dr1_7b"
DEFAULT_AUDIT_DIR = REPO_ROOT / "output" / "dag_audit"
DEFAULT_ASSET_DIR = REPO_ROOT / "docs" / "assets" / "dag_cases"
DEFAULT_OUT = REPO_ROOT / "docs" / "dag_showcase.html"

NON_IMPLICIT_DEP_TYPES = {
    "expr_ref", "expr_overlap", "claim_ref", "var_ref",
    "llm_semantic", "llm_subgoal",
}


def _truncate(text: str, n: int = 220) -> str:
    text = (text or "").replace("\n", " ").strip()
    if len(text) <= n:
        return text
    return text[: n - 1] + "…"


def _score_trace_richness(cached: dict[str, Any]) -> tuple[int, int, int]:
    nodes = cached.get("nodes", []) or []
    edges = cached.get("edges", []) or []
    non_impl = sum(1 for e in edges if str(e.get("dep_type", "")) in NON_IMPLICIT_DEP_TYPES)
    typed = sum(1 for n in nodes if str(n.get("step_type", "")) not in {"", "unknown"})
    return (non_impl, typed, len(nodes))


def _load_cases(cached_path: Path, n: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not cached_path.is_file():
        return rows
    with cached_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    rows = [r for r in rows if isinstance(r.get("cached_dag"), dict) and r["cached_dag"].get("nodes")]
    rows.sort(key=lambda r: _score_trace_richness(r["cached_dag"]), reverse=True)
    return rows[:n]


def _format_dep_hist(edges: list[dict[str, Any]]) -> str:
    counter: Counter[str] = Counter(str(e.get("dep_type", "")) for e in edges)
    items = sorted(counter.items(), key=lambda kv: kv[0])
    chips = []
    for k, v in items:
        chips.append(f'<span class="chip">{html.escape(k)}: {v}</span>')
    return "".join(chips)


def _node_table(nodes: list[dict[str, Any]]) -> str:
    rows = []
    rows.append(
        "<thead><tr><th>step</th><th>type</th><th>verdict</th><th>exprs</th>"
        "<th>claims</th><th>raw_text</th></tr></thead><tbody>"
    )
    for n in nodes:
        sid = n.get("step_id", "")
        stype = html.escape(str(n.get("step_type", "")))
        verdict = html.escape(str(n.get("local_verdict", "")))
        exprs = ", ".join(html.escape(str(e)) for e in (n.get("exprs") or [])[:6])
        claims = ", ".join(html.escape(str(c)) for c in (n.get("claims") or [])[:3])
        raw = html.escape(_truncate(n.get("raw_text", ""), 160))
        rows.append(
            f"<tr><td>{sid}</td><td>{stype}</td><td>{verdict}</td>"
            f"<td>{exprs}</td><td>{claims}</td><td>{raw}</td></tr>"
        )
    rows.append("</tbody>")
    return "".join(rows)


def _edge_table(edges: list[dict[str, Any]]) -> str:
    rows = []
    rows.append(
        "<thead><tr><th>src</th><th>tgt</th><th>edge_type</th><th>dep_type</th>"
        "<th>source_kind</th><th>weight</th><th>evidence</th></tr></thead><tbody>"
    )
    for e in edges:
        src = e.get("source", "")
        tgt = e.get("target", "")
        edge_type = html.escape(str(e.get("edge_type", "")))
        dep_type = html.escape(str(e.get("dep_type", "")))
        source_kind = html.escape(str(e.get("source_kind", "")))
        weight = e.get("weight", 0.0)
        try:
            weight_str = f"{float(weight):.2f}"
        except (TypeError, ValueError):
            weight_str = "-"
        evidence = html.escape(_truncate(e.get("evidence", ""), 200))
        rows.append(
            f"<tr><td>{src}</td><td>{tgt}</td><td>{edge_type}</td>"
            f"<td>{dep_type}</td><td>{source_kind}</td><td>{weight_str}</td>"
            f"<td>{evidence}</td></tr>"
        )
    rows.append("</tbody>")
    return "".join(rows)


_DEP_COLORS = {
    "expr_ref": "#2e6fb5",
    "expr_overlap": "#0ea5e9",
    "claim_ref": "#7c3aed",
    "var_ref": "#10b981",
    "llm_semantic": "#f59e0b",
    "llm_subgoal": "#d97706",
    "implicit_block": "#94a3b8",
    "order": "#cbd5e1",
}


def _safe_filename(text: str, max_len: int = 60) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_]+", "_", text or "case")
    return (cleaned[:max_len] or "case").strip("_") or "case"


def _render_case_png(record: dict[str, Any], out_path: Path) -> bool:
    cached = record.get("cached_dag", {}) or {}
    nodes = cached.get("nodes", []) or []
    edges = cached.get("edges", []) or []
    if not nodes:
        return False

    g = nx.DiGraph()
    valid_ids = set()
    for n in nodes:
        try:
            sid = int(n.get("step_id", 0))
        except (TypeError, ValueError):
            continue
        valid_ids.add(sid)
        g.add_node(sid, step_type=str(n.get("step_type", "")))
    edge_styles = []
    for e in edges:
        try:
            u = int(e.get("source"))
            v = int(e.get("target"))
        except (TypeError, ValueError):
            continue
        if u not in valid_ids or v not in valid_ids:
            continue
        dep = str(e.get("dep_type", ""))
        g.add_edge(u, v, dep_type=dep)
        edge_styles.append((u, v, dep))

    try:
        generations = list(nx.topological_generations(g))
    except nx.NetworkXUnfeasible:
        generations = [list(g.nodes())]
    pos: dict[int, tuple[float, float]] = {}
    for li, layer in enumerate(generations):
        layer_sorted = sorted(layer)
        count = max(1, len(layer_sorted))
        for i, nid in enumerate(layer_sorted):
            pos[nid] = (float(li), (i - (count - 1) / 2.0) * 1.0)
    for nid in g.nodes():
        pos.setdefault(nid, (0.0, 0.0))

    fig, ax = plt.subplots(figsize=(7.0, max(2.4, 0.32 * len(nodes))))
    nx.draw_networkx_nodes(
        g, pos,
        node_color="#bfdbfe",
        edgecolors="#1e3a8a",
        linewidths=1.0,
        node_size=600,
        ax=ax,
    )
    nx.draw_networkx_labels(
        g, pos,
        labels={n: f"S{n}" for n in g.nodes()},
        font_size=8,
        font_color="#0f172a",
        ax=ax,
    )
    by_dep: dict[str, list[tuple[int, int]]] = {}
    for u, v, dep in edge_styles:
        by_dep.setdefault(dep, []).append((u, v))
    for dep, lst in by_dep.items():
        nx.draw_networkx_edges(
            g, pos,
            edgelist=lst,
            edge_color=_DEP_COLORS.get(dep, "#94a3b8"),
            width=1.1 if dep != "order" else 0.7,
            style="solid" if dep != "order" else "dashed",
            arrows=True,
            arrowsize=10,
            ax=ax,
        )
    ax.set_axis_off()
    ax.set_title(
        f"{len(nodes)} nodes / {len(edges)} edges",
        fontsize=9,
        loc="left",
        pad=4,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)
    return True


def _ensure_case_image(
    bench: str,
    case_idx: int,
    record: dict[str, Any],
    asset_dir: Path,
) -> Path | None:
    name = _safe_filename(f"showcase_{bench}_{case_idx}_{(record.get('question') or '')[:32]}")
    out_path = asset_dir / bench / f"{name}.png"
    ok = _render_case_png(record, out_path)
    return out_path if ok else None


def _render_case(idx: int, bench: str, record: dict[str, Any], asset_dir: Path, out_root: Path) -> str:
    cached = record.get("cached_dag", {})
    nodes = cached.get("nodes", []) or []
    edges = cached.get("edges", []) or []
    summary = cached.get("summary", {}) or {}
    question = record.get("question", "")
    gold = record.get("gold", "")
    pred = record.get("pred_pass1", "")
    correct = bool(record.get("correct_pass1"))

    img_path = _ensure_case_image(bench, idx, record, asset_dir)
    img_html = ""
    if img_path:
        try:
            rel = img_path.resolve().relative_to(out_root.resolve())
        except ValueError:
            rel = img_path
        img_html = f'<img src="{rel}" alt="dag_case" />'

    response = _truncate(record.get("response", ""), 1400)
    dep_hist_html = _format_dep_hist(edges)

    return f"""
    <article class="case">
      <header class="case-head">
        <span class="bench-tag">{html.escape(bench)}</span>
        <span class="case-id">case {idx + 1}</span>
        <span class="status {'ok' if correct else 'fail'}">
          {'✓ correct@1' if correct else '✗ incorrect@1'}
        </span>
      </header>
      <div class="case-body">
        <div class="case-left">
          <div class="kv"><span class="k">Question</span><span class="v">{html.escape(_truncate(question, 320))}</span></div>
          <div class="kv"><span class="k">Gold</span><span class="v">{html.escape(str(gold))}</span></div>
          <div class="kv"><span class="k">Pred</span><span class="v">{html.escape(str(pred))}</span></div>
          <div class="kv"><span class="k">DAG</span><span class="v">{summary.get('num_nodes', len(nodes))} nodes / {summary.get('num_edges', len(edges))} edges</span></div>
          <div class="dep-hist">{dep_hist_html}</div>
          <details>
            <summary>response (truncated)</summary>
            <pre class="resp">{html.escape(response)}</pre>
          </details>
        </div>
        <div class="case-right">{img_html}</div>
      </div>
      <details class="tables" open>
        <summary>Nodes &amp; edges</summary>
        <h4>Nodes</h4>
        <table class="data">{_node_table(nodes)}</table>
        <h4>Edges</h4>
        <table class="data">{_edge_table(edges)}</table>
      </details>
    </article>
    """


CSS = """
:root {
  --bg: #f7f8fb; --panel: #ffffff; --ink: #1f2937; --muted: #6b7280;
  --accent: #2e6fb5; --border: #e5e7eb;
}
body { margin: 0; background: var(--bg); color: var(--ink);
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Helvetica Neue',
    Arial, 'Noto Sans', 'PingFang SC', 'Microsoft YaHei', sans-serif;
  line-height: 1.55; }
header.page {
  background: linear-gradient(120deg, #1e3a8a, #2e6fb5); color: white;
  padding: 26px 36px; box-shadow: 0 2px 8px rgba(0,0,0,0.08);
}
header.page h1 { margin: 0 0 4px 0; font-size: 24px; }
header.page .sub { opacity: 0.85; font-size: 13px; }
main { max-width: 1240px; margin: 24px auto 60px auto; padding: 0 24px; }
nav.benches {
  position: sticky; top: 0; background: var(--panel); border: 1px solid var(--border);
  border-radius: 8px; padding: 8px 12px; display: flex; gap: 8px; flex-wrap: wrap;
  z-index: 5; box-shadow: 0 1px 3px rgba(0,0,0,0.05);
}
nav.benches a { color: var(--accent); text-decoration: none; font-weight: 500; padding: 4px 8px; border-radius: 4px; font-size: 13px; }
nav.benches a:hover { background: #eef2ff; }
section.bench { margin-top: 22px; padding: 18px; background: var(--panel); border: 1px solid var(--border); border-radius: 10px; }
section.bench > h2 { margin-top: 0; color: var(--accent); border-bottom: 2px solid var(--border); padding-bottom: 8px; }
.case { margin-top: 16px; border: 1px solid var(--border); border-radius: 10px; padding: 14px 16px; background: #fafbfc; }
.case-head { display: flex; gap: 10px; align-items: center; margin-bottom: 8px; }
.bench-tag { background: #2e6fb5; color: white; padding: 2px 8px; border-radius: 999px; font-size: 11px; }
.case-id { color: var(--muted); font-size: 12px; }
.status { font-size: 12px; padding: 2px 8px; border-radius: 999px; font-weight: 600; }
.status.ok { background: #d1fae5; color: #065f46; }
.status.fail { background: #fee2e2; color: #991b1b; }
.case-body { display: grid; grid-template-columns: 1.05fr 0.95fr; gap: 16px; align-items: start; }
.case-left .kv { display: grid; grid-template-columns: 80px 1fr; gap: 6px; font-size: 13px; margin-bottom: 4px; }
.case-left .kv .k { color: var(--muted); font-weight: 500; }
.case-left .kv .v { color: var(--ink); }
.dep-hist { margin-top: 8px; }
.chip { display: inline-block; background: #eef2ff; color: #1e3a8a; padding: 2px 8px; border-radius: 999px; font-size: 11px; margin: 2px 4px 2px 0; font-family: ui-monospace, Menlo, Consolas, monospace; }
pre.resp { background: #0f172a; color: #e2e8f0; padding: 8px 10px; border-radius: 6px; max-height: 220px; overflow: auto; font-size: 12px; white-space: pre-wrap; word-break: break-word; }
.case-right img { max-width: 100%; border: 1px solid var(--border); border-radius: 6px; background: white; }
details.tables { margin-top: 14px; }
details.tables summary { font-weight: 600; color: var(--ink); cursor: pointer; }
table.data { width: 100%; border-collapse: collapse; font-size: 12.5px; margin-top: 6px; }
table.data th, table.data td { border-bottom: 1px solid var(--border); padding: 4px 8px; vertical-align: top; }
table.data th { background: #f3f6fb; text-align: left; }
.footer { color: var(--muted); font-size: 12px; margin: 26px 16px 0 16px; }
.footer a { color: var(--accent); }
"""


def main() -> int:
    parser = argparse.ArgumentParser(description="Build docs/dag_showcase.html.")
    parser.add_argument("--label", default=DEFAULT_LABEL)
    parser.add_argument("--audit-dir", type=Path, default=DEFAULT_AUDIT_DIR)
    parser.add_argument("--asset-dir", type=Path, default=DEFAULT_ASSET_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--n", type=int, default=3, help="Cases per benchmark.")
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        default=[
            "gsm8k", "math500", "olympiadbench", "omni_math",
            "aime2024", "aime2025", "cnmo2024", "mmlu", "gpqa_diamond",
        ],
    )
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    out_root = args.out.parent

    nav_links = []
    sections = []
    for bench in args.benchmarks:
        cached_path = args.audit_dir / f"{args.label}_{bench}_cached.jsonl"
        cases = _load_cases(cached_path, args.n)
        if not cases:
            continue
        nav_links.append(f'<a href="#bench-{bench}">{bench}</a>')
        case_html = "".join(_render_case(i, bench, c, args.asset_dir, out_root) for i, c in enumerate(cases))
        sections.append(
            f'<section class="bench" id="bench-{bench}">\n'
            f'  <h2>{bench} <small style="color:var(--muted); font-weight:400;">'
            f'({len(cases)} cases)</small></h2>\n'
            f'{case_html}\n'
            f'</section>'
        )

    if not sections:
        print("[warn] no benchmarks had cached jsonl + cases; emitting empty page.")
        sections.append('<section class="bench"><p>No DAG cases found.</p></section>')

    html_out = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<title>TopoPRM — DAG Showcase</title>
<style>{CSS}</style>
</head>
<body>
<header class="page">
  <h1>TopoPRM — DAG Showcase</h1>
  <div class="sub">Per-benchmark cached DAGs from <code>output/dag_audit/</code>. Auto-generated by <code>scripts/build_dag_showcase_html.py</code>.</div>
</header>
<main>
  <nav class="benches">{''.join(nav_links)}</nav>
  {''.join(sections)}
  <p class="footer">See <a href="tgsd_framework.html">tgsd_framework.html</a> for the full TGSD pipeline overview.</p>
</main>
</body>
</html>
"""
    args.out.write_text(html_out, encoding="utf-8")
    print(f"[ok] wrote {args.out} with {len(nav_links)} benchmark sections")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

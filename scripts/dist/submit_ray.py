#!/usr/bin/env python3
"""submit_ray.py

Ray-based dispatcher for TopoPRM evaluation tasks. Portable across managed
Kubernetes / Ray clusters and elastic resource pools, as well as any
self-hosted Ray deployment.

Workflow:
    1. Read the manifest (TSV) and pick the TASK_IDs requested via --task-ids
       (default: every row in the manifest).
    2. Submit each TASK_ID as a Ray remote function with `num_gpus=1`. Ray
       schedules them onto worker nodes that have a free GPU.
    3. Each remote function calls `scripts/dist/run_eval_worker.sh <TASK_ID>`
       in a subprocess, exporting RAY_TASK_INDEX so the worker contract is
       satisfied.

Examples:
    # Local dry-run on the head node (works on a single-node Ray instance):
    python3 scripts/dist/submit_ray.py --address auto --task-ids 0..5

    # Submit to a remote Ray cluster:
    RAY_ADDRESS=ray://${RAY_HEAD_HOST}:10001 \
        python3 scripts/dist/submit_ray.py --task-ids 0..9
"""
from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path


def _expand_ranges(tokens: list[str]) -> list[int]:
    out: list[int] = []
    for t in tokens:
        if ".." in t:
            a, b = t.split("..", 1)
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(t))
    return out


def _read_manifest_ids(manifest: Path) -> list[int]:
    ids: list[int] = []
    for line in manifest.read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        fields = line.split("\t")
        if len(fields) < 7:
            continue
        try:
            ids.append(int(fields[0]))
        except ValueError:
            continue
    return ids


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--address", default=os.environ.get("RAY_ADDRESS", "auto"),
                        help="Ray cluster address. 'auto' attaches to a local Ray "
                             "instance; use 'ray://host:port' for a remote cluster.")
    parser.add_argument("--manifest", default="configs/dist/eval_manifest.tsv")
    parser.add_argument("--task-ids", nargs="*", default=None,
                        help="Subset of TASK_IDs to submit. Accepts integers or "
                             "N..M ranges. Defaults to every row in the manifest.")
    parser.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[2]),
                        help="Absolute path to the TopoPRM repo root on the cluster nodes.")
    parser.add_argument("--num-gpus-per-task", type=float, default=1.0)
    parser.add_argument("--num-cpus-per-task", type=float, default=4.0)
    parser.add_argument("--timeout-sec", type=int, default=0,
                        help="Kill an individual task after this many seconds (0 = no timeout).")
    args = parser.parse_args()

    try:
        import ray  # noqa: WPS433  (import inside main keeps the CLI usable when ray is absent)
    except ImportError:
        print("ray is not installed in this environment; please `pip install ray`.",
              file=sys.stderr)
        return 1

    manifest_path = Path(args.repo_root) / args.manifest
    if not manifest_path.exists():
        print(f"manifest not found: {manifest_path}", file=sys.stderr)
        return 1

    all_ids = _read_manifest_ids(manifest_path)
    if args.task_ids:
        task_ids = _expand_ranges(args.task_ids)
        unknown = [t for t in task_ids if t not in all_ids]
        if unknown:
            print(f"WARNING: task ids not in manifest: {unknown}", file=sys.stderr)
    else:
        task_ids = all_ids
    if not task_ids:
        print("no task ids to submit", file=sys.stderr)
        return 1

    print(f"[ray] address={args.address}  tasks={task_ids}")
    ray.init(address=args.address, ignore_reinit_error=True)

    @ray.remote(num_gpus=args.num_gpus_per_task, num_cpus=args.num_cpus_per_task)
    def _run(task_id: int, repo_root: str, timeout_sec: int) -> dict:
        env = os.environ.copy()
        env["RAY_TASK_INDEX"] = str(task_id)
        env["TASK_ID"] = str(task_id)
        env.setdefault("NUM_GPUS_PER_NODE", "1")
        cmd = ["bash", "scripts/dist/run_eval_worker.sh", str(task_id)]
        t0 = time.time()
        try:
            proc = subprocess.run(
                cmd,
                cwd=repo_root,
                env=env,
                timeout=timeout_sec if timeout_sec > 0 else None,
                check=False,
                capture_output=True,
                text=True,
            )
            return {
                "task_id": task_id,
                "returncode": proc.returncode,
                "elapsed_sec": round(time.time() - t0, 1),
                "stdout_tail": (proc.stdout or "")[-1500:],
                "stderr_tail": (proc.stderr or "")[-1500:],
            }
        except subprocess.TimeoutExpired:
            return {
                "task_id": task_id,
                "returncode": -1,
                "elapsed_sec": round(time.time() - t0, 1),
                "stdout_tail": "",
                "stderr_tail": f"timeout after {timeout_sec}s",
            }

    futures = [
        _run.remote(tid, args.repo_root, args.timeout_sec)
        for tid in task_ids
    ]
    results = ray.get(futures)

    failed: list[dict] = []
    for r in results:
        marker = "OK " if r["returncode"] == 0 else "FAIL"
        print(f"[ray] {marker} task={r['task_id']:>3}  rc={r['returncode']}  "
              f"elapsed={r['elapsed_sec']}s")
        if r["returncode"] != 0:
            failed.append(r)
    if failed:
        print(f"[ray] {len(failed)}/{len(results)} task(s) failed:")
        for r in failed:
            print(f"  --- task {r['task_id']} stderr tail ---\n{r['stderr_tail']}")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Unified 9-benchmark evaluation orchestrator.

Schedules `scripts/bench_transformers.py` across free GPUs, one benchmark per
worker, with priority-aware grouping (fast benches first), automatic retry,
streaming logs, wandb monitoring, and a live metrics summary.

Design goals:
  1. One-command entry: pick model + adapter, get 9 benchmarks on all free GPUs.
  2. Prioritize quick-return benches (GSM8K/MATH-500/AIME/CNMO) before slow
     benches (MMLU, GPQA-D, Olympiad, Omni-MATH) so the paper table gets at
     least *some* real numbers even if a slow bench later fails.
  3. Isolate known-flaky benches: MMLU runs with --max_items subset; any
     benchmark already complete (metrics.json exists) is skipped unless
     --force.
  4. Recover from transient CUDA/OOM faults via single automatic retry with
     reduced batch size.

Usage:
    python scripts/unified_eval_orchestrator.py \
        --model ${HF_MODELS_DIR:-./models}/Qwen/Qwen3.5-9B \
        --label qwen35_9b_base \
        --gpus 0,1,2,3 \
        --benchmarks all
"""
from __future__ import annotations

import argparse
import json
import os
import queue
import re
import shlex
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional


ALL_BENCHMARKS = [
    "gsm8k",
    "math500",
    "aime2024",
    "aime2025",
    "cnmo2024",
    "olympiadbench",
    "omni_math",
    "gpqa_diamond",
    "mmlu",
]

FAST_BENCHES = {"gsm8k", "math500", "aime2024", "aime2025", "cnmo2024"}
SLOW_BENCHES = {"olympiadbench", "omni_math", "gpqa_diamond", "mmlu"}

BENCH_CONFIG = {
    "gsm8k":          {"max_new_tokens": 4096, "max_items": 0,   "batch_size": 2, "est_min": 90},
    "math500":        {"max_new_tokens": 4096, "max_items": 0,   "batch_size": 2, "est_min": 90},
    "aime2024":       {"max_new_tokens": 4096, "max_items": 0,   "batch_size": 2, "est_min": 25},
    "aime2025":       {"max_new_tokens": 4096, "max_items": 0,   "batch_size": 2, "est_min": 25},
    "cnmo2024":       {"max_new_tokens": 4096, "max_items": 0,   "batch_size": 2, "est_min": 40},
    "olympiadbench":  {"max_new_tokens": 4096, "max_items": 500, "batch_size": 2, "est_min": 120},
    "omni_math":      {"max_new_tokens": 4096, "max_items": 500, "batch_size": 2, "est_min": 150},
    "gpqa_diamond":   {"max_new_tokens": 4096, "max_items": 0,   "batch_size": 2, "est_min": 60},
    "mmlu":           {"max_new_tokens": 4096, "max_items": 1500,"batch_size": 2, "est_min": 200},
}


@dataclass
class Task:
    bench: str
    model: str
    adapter: str
    label: str
    use_chat_template: bool
    sft_style: bool
    num_samples_per_item: int
    k_values: list[int]
    output_dir: str
    log_dir: str
    extra_flags: list[str] = field(default_factory=list)
    attempt: int = 0
    max_attempts: int = 2

    @property
    def metrics_path(self) -> Path:
        return Path(self.output_dir) / f"{self.label}_{self.bench}_metrics.json"

    @property
    def log_path(self) -> Path:
        return Path(self.log_dir) / f"{self.label}_{self.bench}.log"

    def cmd(self, gpu_id: int, batch_size: int, max_new_tokens: int, max_items: int) -> list[str]:
        repo_root = Path(__file__).resolve().parents[1]
        python_bin = os.environ.get(
            "TOPOPRM_PYTHON", "${TOPOPRM_PYTHON:-python3}"
        )
        if not Path(python_bin).exists():
            python_bin = sys.executable
        cmd = [
            python_bin, "-u", str(repo_root / "scripts" / "bench_transformers.py"),
            "--model", self.model,
            "--label", self.label,
            "--benchmarks", self.bench,
            "--batch_size", str(batch_size),
            "--max_new_tokens", str(max_new_tokens),
            "--num_samples_per_item", str(self.num_samples_per_item),
            "--k_values", *[str(k) for k in self.k_values],
            "--output_dir", self.output_dir,
        ]
        if max_items > 0:
            cmd += ["--max_items", str(max_items)]
        if self.use_chat_template:
            cmd.append("--use_chat_template")
        if self.sft_style:
            cmd.append("--sft_style")
            if self.bench == "mmlu":
                cmd.append("--allow_mmlu_sft_style")
        cmd += self.extra_flags
        return cmd


@dataclass
class WorkerResult:
    task: Task
    gpu_id: int
    exit_code: int
    duration_sec: float
    pass_at_1: Optional[float]


class Orchestrator:
    def __init__(self, args):
        self.args = args
        self.repo_root = Path(__file__).resolve().parents[1]
        self.log_dir = Path(args.log_dir)
        self.output_dir = Path(args.output_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.run_log = self.log_dir / f"orchestrator_{args.label}_{datetime.now():%Y%m%d_%H%M%S}.log"
        self.status_json = self.log_dir / f"status_{args.label}.json"
        self.gpu_queue: queue.Queue[int] = queue.Queue()
        for g in args.gpus:
            self.gpu_queue.put(g)
        self.lock = threading.Lock()
        self.results: list[WorkerResult] = []
        self.running: dict[int, Task] = {}
        self.stop_event = threading.Event()

    def log(self, msg: str):
        ts = datetime.now().strftime("%H:%M:%S")
        line = f"[{ts}] {msg}"
        print(line, flush=True)
        with self.run_log.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    def build_tasks(self) -> list[Task]:
        if self.args.benchmarks == ["all"]:
            benches = list(ALL_BENCHMARKS)
        else:
            benches = list(self.args.benchmarks)
        order_fast = [b for b in benches if b in FAST_BENCHES]
        order_slow = [b for b in benches if b in SLOW_BENCHES]
        # fast first so the paper table gets real numbers ASAP
        ordered = order_fast + order_slow
        tasks = []
        for b in ordered:
            t = Task(
                bench=b,
                model=self.args.model,
                adapter=self.args.adapter,
                label=self.args.label,
                use_chat_template=self.args.use_chat_template,
                sft_style=self.args.sft_style,
                num_samples_per_item=self.args.num_samples_per_item,
                k_values=self.args.k_values,
                output_dir=str(self.output_dir),
                log_dir=str(self.log_dir),
            )
            if self.args.adapter:
                t.extra_flags += ["--adapter", self.args.adapter]
            if self.args.force:
                t.extra_flags.append("--force_overwrite")
            if self.args.save_solutions:
                t.extra_flags.append("--save_solutions")
            tasks.append(t)
        return tasks

    def prune_completed(self, tasks: list[Task]) -> list[Task]:
        pending = []
        for t in tasks:
            if not self.args.force and t.metrics_path.exists():
                try:
                    m = json.loads(t.metrics_path.read_text())
                    p1 = m.get("pass@1") or m.get("accuracy") or 0
                    self.log(f"SKIP {t.bench}: metrics.json exists pass@1={p1:.3f}")
                    continue
                except Exception:
                    pass
            pending.append(t)
        return pending

    def run_task(self, task: Task, gpu_id: int) -> WorkerResult:
        attempt_suffix = "" if task.attempt == 0 else f"_retry{task.attempt}"
        log_path = Path(task.log_dir) / f"{task.label}_{task.bench}{attempt_suffix}.log"
        cfg = BENCH_CONFIG[task.bench]
        batch_size = max(1, cfg["batch_size"] // (2 ** task.attempt))
        max_new_tokens = cfg["max_new_tokens"]
        max_items = cfg["max_items"]
        cmd = task.cmd(gpu_id, batch_size, max_new_tokens, max_items)
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        env.setdefault("PYTHONPATH", str(self.repo_root))
        env["PYTHONPATH"] = f"{self.repo_root}:{env.get('PYTHONPATH','')}"
        env["TOKENIZERS_PARALLELISM"] = "false"

        start = time.time()
        header = (
            f"==> {task.bench} | label={task.label} | GPU={gpu_id} "
            f"| attempt={task.attempt+1}/{task.max_attempts} "
            f"| batch={batch_size} mnt={max_new_tokens} max_items={max_items or 'full'} "
            f"| eta≈{cfg['est_min']}min\ncmd: {' '.join(shlex.quote(c) for c in cmd)}\n\n"
        )
        log_path.write_text(header, encoding="utf-8")
        with self.lock:
            self.running[gpu_id] = task
            self._persist_status()

        self.log(
            f"START {task.bench} on GPU {gpu_id} (attempt {task.attempt+1}) "
            f"log -> {log_path}"
        )

        with log_path.open("a", encoding="utf-8") as logf:
            proc = subprocess.Popen(
                cmd,
                env=env,
                cwd=str(self.repo_root),
                stdout=logf,
                stderr=subprocess.STDOUT,
                preexec_fn=os.setsid,
            )
            try:
                while proc.poll() is None:
                    if self.stop_event.is_set():
                        self.log(f"STOP requested; terminating PID {proc.pid}")
                        try:
                            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                        except Exception:
                            pass
                        break
                    time.sleep(3.0)
            except KeyboardInterrupt:
                self.stop_event.set()
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                except Exception:
                    pass
            rc = proc.wait()

        duration = time.time() - start
        pass1 = None
        if task.metrics_path.exists():
            try:
                m = json.loads(task.metrics_path.read_text())
                pass1 = float(m.get("pass@1", m.get("accuracy", 0.0)))
            except Exception:
                pass1 = None
        res = WorkerResult(task=task, gpu_id=gpu_id, exit_code=rc,
                           duration_sec=duration, pass_at_1=pass1)
        status = "OK" if rc == 0 and pass1 is not None else "FAIL"
        p1str = f"{pass1*100:.1f}%" if pass1 is not None else "N/A"
        self.log(
            f"DONE {task.bench} GPU {gpu_id} rc={rc} pass@1={p1str} "
            f"duration={duration/60:.1f}min status={status}"
        )
        with self.lock:
            self.running.pop(gpu_id, None)
            self._persist_status()
        return res

    def _persist_status(self):
        data = {
            "label": self.args.label,
            "model": self.args.model,
            "adapter": self.args.adapter,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "running": {
                str(g): {
                    "bench": t.bench,
                    "attempt": t.attempt + 1,
                    "log": str(Path(t.log_dir) / f"{t.label}_{t.bench}.log"),
                }
                for g, t in self.running.items()
            },
            "completed": [
                {
                    "bench": r.task.bench,
                    "gpu": r.gpu_id,
                    "exit_code": r.exit_code,
                    "pass@1": r.pass_at_1,
                    "duration_min": round(r.duration_sec / 60, 2),
                    "attempt": r.task.attempt + 1,
                }
                for r in self.results
            ],
        }
        self.status_json.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")

    def worker_loop(self, pending: list[Task]):
        pending_q: queue.Queue[Task] = queue.Queue()
        for t in pending:
            pending_q.put(t)

        def worker():
            while not pending_q.empty() and not self.stop_event.is_set():
                try:
                    task = pending_q.get_nowait()
                except queue.Empty:
                    return
                gpu_id = self.gpu_queue.get()
                try:
                    res = self.run_task(task, gpu_id)
                    with self.lock:
                        self.results.append(res)
                    if res.exit_code != 0 and task.attempt + 1 < task.max_attempts:
                        task.attempt += 1
                        self.log(
                            f"RETRY {task.bench} (attempt {task.attempt+1}/"
                            f"{task.max_attempts}) after failure"
                        )
                        pending_q.put(task)
                finally:
                    self.gpu_queue.put(gpu_id)

        threads = [threading.Thread(target=worker, name=f"w{i}", daemon=True)
                   for i in range(len(self.args.gpus))]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

    def summarize(self):
        self.log("=" * 60)
        self.log(f"Summary for label={self.args.label}")
        self.log("=" * 60)
        rows = []
        for b in ALL_BENCHMARKS:
            mp = self.output_dir / f"{self.args.label}_{b}_metrics.json"
            if mp.exists():
                try:
                    m = json.loads(mp.read_text())
                    p1 = float(m.get("pass@1", 0.0))
                    rows.append((b, p1, m.get("elapsed_sec", 0.0), "ok"))
                except Exception:
                    rows.append((b, 0.0, 0.0, "parse_err"))
            else:
                rows.append((b, None, None, "missing"))
        fmt = "  {:<15} {:>8} {:>8} {:>8}"
        self.log(fmt.format("benchmark", "pass@1", "elapsed", "status"))
        for b, p1, elapsed, st in rows:
            p1str = f"{p1*100:.1f}%" if isinstance(p1, float) else "--"
            tstr = f"{elapsed:.0f}s" if isinstance(elapsed, float) and elapsed else "--"
            self.log(fmt.format(b, p1str, tstr, st))

    def run(self):
        tasks = self.build_tasks()
        pending = self.prune_completed(tasks)
        if not pending:
            self.log("Nothing to do; all metrics.json exist. Use --force to re-run.")
            self.summarize()
            return 0
        self.log(
            f"Launching {len(pending)} benches on GPUs {self.args.gpus}: "
            f"{[t.bench for t in pending]}"
        )
        signal.signal(signal.SIGINT, lambda s, f: self.stop_event.set())
        signal.signal(signal.SIGTERM, lambda s, f: self.stop_event.set())
        self.worker_loop(pending)
        self.summarize()
        failed = [r for r in self.results if r.exit_code != 0 and r.pass_at_1 is None]
        return 1 if failed else 0


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True, help="Local HF model path.")
    p.add_argument("--adapter", default="", help="Optional LoRA adapter dir.")
    p.add_argument("--label", required=True, help="Short run label.")
    p.add_argument("--gpus", default="0,1,2,3,4,5,6,7",
                   help="Comma-separated CUDA ids to use.")
    p.add_argument("--benchmarks", nargs="+", default=["all"],
                   help="Which benchmarks to run; 'all' = 9 defaults.")
    p.add_argument("--num_samples_per_item", type=int, default=1)
    p.add_argument("--k_values", nargs="+", type=int, default=[1])
    p.add_argument("--use_chat_template", action="store_true", default=True,
                   help="Use tokenizer chat template (default ON).")
    p.add_argument("--no_chat_template", dest="use_chat_template",
                   action="store_false", help="Disable chat template.")
    p.add_argument("--sft_style", action="store_true",
                   help="Add --sft_style (for SFT/GRPO adapters).")
    p.add_argument("--force", action="store_true",
                   help="Re-run benches even if metrics.json already exists.")
    p.add_argument("--save_solutions", action="store_true")
    p.add_argument("--output_dir", default="output/eval")
    p.add_argument("--log_dir", default="logs/unified")
    args = p.parse_args(argv)
    args.gpus = [int(x) for x in args.gpus.split(",") if x.strip() != ""]
    return args


def main(argv=None):
    args = parse_args(argv)
    orch = Orchestrator(args)
    return orch.run()


if __name__ == "__main__":
    sys.exit(main())

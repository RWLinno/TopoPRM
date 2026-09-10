#!/usr/bin/env python3
"""Unified benchmark evaluation orchestrator.

Schedules `scripts/bench_transformers.py` across free GPUs, one benchmark per
worker, with priority-aware grouping (fast benches first), automatic retry,
streaming logs, wandb monitoring, and a live metrics summary.

Design goals:
  1. One-command entry: pick model + adapter, run all registered benchmarks on
     the available GPUs.
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
        --model ${MODEL_ROOT}/Qwen/Qwen3.5-9B \
        --label qwen35_9b_base \
        --gpus 0,1,2,3 \
        --benchmarks all
"""
from __future__ import annotations

import argparse
import hashlib
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
    "aime2026",
    "cnmo2024",
    "olympiadbench",
    "omni_math",
    "gpqa_diamond",
    "mmlu",
]

FAST_BENCHES = {
    "gsm8k", "math500", "aime2024", "aime2025", "aime2026", "cnmo2024"
}
SLOW_BENCHES = {"olympiadbench", "omni_math", "gpqa_diamond", "mmlu"}

BENCH_CONFIG = {
    "gsm8k":          {"max_new_tokens": 4096, "max_items": 0,   "batch_size": 4, "est_min": 45},
    "math500":        {"max_new_tokens": 4096, "max_items": 0,   "batch_size": 4, "est_min": 60},
    "aime2024":       {"max_new_tokens": 4096, "max_items": 0,   "batch_size": 2, "est_min": 25},
    "aime2025":       {"max_new_tokens": 4096, "max_items": 0,   "batch_size": 2, "est_min": 25},
    "aime2026":       {"max_new_tokens": 4096, "max_items": 0,   "batch_size": 2, "est_min": 25},
    "cnmo2024":       {"max_new_tokens": 4096, "max_items": 0,   "batch_size": 2, "est_min": 40},
    "olympiadbench":  {"max_new_tokens": 4096, "max_items": 0,   "batch_size": 2, "est_min": 120},
    "omni_math":      {"max_new_tokens": 4096, "max_items": 500, "batch_size": 2, "est_min": 150},
    "gpqa_diamond":   {"max_new_tokens": 1536, "max_items": 0,   "batch_size": 4, "est_min": 35},
    "mmlu":           {"max_new_tokens": 768,  "max_items": 1500,"batch_size": 8, "est_min": 100},
}

_LOCAL_BENCHMARK_FILES = {
    "gsm8k": "data/benchmarks/GSM8K/test.jsonl",
    "math500": "data/benchmarks/MATH-500/test.jsonl",
    "aime2024": "data/benchmarks/AIME2024/train.jsonl",
    "aime2025": "data/benchmarks/AIME2025/train.jsonl",
    "aime2026": "data/benchmarks/AIME2026/train.jsonl",
    "olympiadbench": "data/benchmarks/OlympiadBench/test_en_oe_to_math.jsonl",
    "gpqa_diamond": "data/benchmarks/GPQA_Diamond/test.jsonl",
    "mmlu": "data/benchmarks/MMLU/test.jsonl",
}

_MATH_RESPONSE_BENCHMARKS = {
    "gsm8k",
    "math500",
    "olympiadbench",
    "omni_math",
    "aime2024",
    "aime2025",
    "aime2026",
    "cnmo2024",
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _rescoring_source_fingerprints() -> dict[str, str]:
    root = Path(__file__).resolve().parents[1]
    sources = (
        root / "scripts" / "bench_transformers.py",
        root / "src" / "eval" / "unified_benchmark.py",
        root / "src" / "eval" / "math_scoring.py",
        root / "src" / "reward" / "outcome_reward.py",
    )
    return {str(path.relative_to(root)): _sha256_file(path) for path in sources}


def _benchmark_source_fingerprint(bench: str) -> dict:
    relative = _LOCAL_BENCHMARK_FILES.get(bench)
    if relative is None:
        return {"kind": "registered_loader", "benchmark": bench}
    path = Path(__file__).resolve().parents[1] / relative
    if not path.is_file():
        return {"kind": "registered_loader", "benchmark": bench, "local_path": relative}
    digest = hashlib.sha256()
    rows = 0
    with path.open("rb") as handle:
        for line in handle:
            digest.update(line)
            rows += bool(line.strip())
    return {
        "kind": "local_jsonl",
        "path": relative,
        "rows": rows,
        "bytes": path.stat().st_size,
        "sha256": digest.hexdigest(),
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
    pass1_do_sample: bool
    temperature: float
    top_p: float
    top_k: int
    min_p: float
    repetition_penalty: float
    eval_seed: int
    fold_system_into_user: bool
    force_think_prefix: bool
    system_control: str
    empty_system_prompt: bool
    user_suffix: str
    score_topology: bool
    output_dir: str
    log_dir: str
    extra_flags: list[str] = field(default_factory=list)
    attempt: int = 0
    max_attempts: int = 2

    @property
    def metrics_path(self) -> Path:
        return Path(self.output_dir) / f"{self.label}_{self.bench}_metrics.json"

    @property
    def details_path(self) -> Path:
        return Path(self.output_dir) / f"{self.label}_{self.bench}_details.jsonl"

    @property
    def log_path(self) -> Path:
        return Path(self.log_dir) / f"{self.label}_{self.bench}.log"

    def cmd(self, gpu_id: int, batch_size: int, max_new_tokens: int, max_items: int) -> list[str]:
        repo_root = Path(__file__).resolve().parents[1]
        python_bin = os.environ.get(
            "TOPOPRM_PYTHON", os.path.expandvars("${PYTHON_ENV_BIN}/python")
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
            "--temperature", str(self.temperature),
            "--top_p", str(self.top_p),
            "--top_k", str(self.top_k),
            "--min_p", str(self.min_p),
            "--repetition_penalty", str(self.repetition_penalty),
            "--eval_seed", str(self.eval_seed),
            "--output_dir", self.output_dir,
        ]
        if max_items > 0:
            cmd += ["--max_items", str(max_items)]
        if self.use_chat_template:
            cmd.append("--use_chat_template")
        if self.pass1_do_sample:
            cmd.append("--pass1_do_sample")
        if self.fold_system_into_user:
            cmd.append("--fold_system_into_user")
        if self.force_think_prefix:
            cmd.append("--force_think_prefix")
        if self.system_control:
            cmd += ["--system_control", self.system_control]
        if self.empty_system_prompt:
            cmd.append("--empty_system_prompt")
        if self.user_suffix:
            cmd += ["--user_suffix", self.user_suffix]
        if self.score_topology:
            cmd.append("--score_topology")
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
                pass1_do_sample=self.args.pass1_do_sample,
                temperature=self.args.temperature,
                top_p=self.args.top_p,
                top_k=self.args.top_k,
                min_p=self.args.min_p,
                repetition_penalty=self.args.repetition_penalty,
                eval_seed=self.args.eval_seed,
                fold_system_into_user=self.args.fold_system_into_user,
                force_think_prefix=self.args.force_think_prefix,
                system_control=self.args.system_control,
                empty_system_prompt=self.args.empty_system_prompt,
                user_suffix=self.args.user_suffix,
                score_topology=self.args.score_topology,
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
                    expected = {
                        "label": t.label,
                        "num_samples_per_item": t.num_samples_per_item,
                        "k_values": t.k_values,
                        "use_chat_template": t.use_chat_template,
                        "fold_system_into_user": t.fold_system_into_user,
                        "force_think_prefix": t.force_think_prefix,
                        "system_control": t.system_control,
                        "empty_system_prompt": t.empty_system_prompt,
                        "user_suffix": t.user_suffix,
                        "sft_style": t.sft_style,
                        "model": str(Path(t.model).resolve()),
                        "adapter": str(Path(t.adapter).resolve()) if t.adapter else "",
                        "max_new_tokens": BENCH_CONFIG[t.bench]["max_new_tokens"],
                        "batch_size": BENCH_CONFIG[t.bench]["batch_size"],
                        "pass1_do_sample": t.pass1_do_sample,
                        "temperature": t.temperature,
                        "top_p": t.top_p,
                        "top_k": t.top_k,
                        "min_p": t.min_p,
                        "repetition_penalty": t.repetition_penalty,
                        "eval_seed": t.eval_seed,
                        "score_topology": t.score_topology,
                        "bootstrap_n_resamples": 10_000,
                        "bootstrap_seed": 0,
                        "pass1_decoding": (
                            "sampled" if t.pass1_do_sample else "greedy"
                        ),
                        "provenance_schema_version": 3,
                        "visible_token_counting": (
                            "retokenized_decoded_response_no_special_tokens"
                        ),
                    }
                    protocol_matches = True
                    for key, value in expected.items():
                        if key == "empty_system_prompt":
                            observed = m.get(key, False)
                        elif key == "user_suffix":
                            observed = m.get(key, "")
                        else:
                            if key not in m:
                                protocol_matches = False
                                break
                            observed = m[key]
                        if observed != value:
                            protocol_matches = False
                            break
                    source_matches = m.get("provenance", {}).get(
                        "benchmark_source"
                    ) == _benchmark_source_fingerprint(t.bench)
                    strict_math_matches = (
                        t.bench not in _MATH_RESPONSE_BENCHMARKS
                        or m.get("math_scoring_protocol")
                        == "last_nonempty_box_else_explicit_final_answer_math_verify_gold_first"
                    )
                    rescoring_matches = m.get("provenance", {}).get(
                        "rescoring_sources"
                    ) == _rescoring_source_fingerprints()
                    if (
                        protocol_matches
                        and source_matches
                        and strict_math_matches
                        and rescoring_matches
                    ):
                        p1 = m.get("pass@1") or m.get("accuracy") or 0
                        self.log(
                            f"SKIP {t.bench}: compatible metrics exist pass@1={p1:.3f}"
                        )
                        continue
                    self.log(f"RERUN {t.bench}: existing metrics use another protocol")
                    t.extra_flags.append("--force_overwrite")
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

        canonical_with_details = (
            task.num_samples_per_item == 1
            and task.k_values == [1]
            and "--save_solutions" in task.extra_flags
        )
        if rc == 0 and canonical_with_details:
            if not task.metrics_path.is_file() or not task.details_path.is_file():
                self.log(
                    f"RESCORE FAIL {task.bench}: canonical run did not emit both "
                    "metrics and per-item details"
                )
                rc = 3
            else:
                python_bin = os.environ.get(
                    "TOPOPRM_PYTHON",
                    os.path.expandvars("${PYTHON_ENV_BIN}/python"),
                )
                if not Path(python_bin).exists():
                    python_bin = sys.executable
                rescore_cmd = [
                    python_bin,
                    "-u",
                    str(self.repo_root / "scripts" / "rescore_eval_details.py"),
                    "--workers",
                    "8",
                    str(task.details_path),
                ]
                self.log(f"RESCORE {task.bench}: strict canonical post-pass")
                with log_path.open("a", encoding="utf-8") as logf:
                    completed = subprocess.run(
                        rescore_cmd,
                        env=env,
                        cwd=str(self.repo_root),
                        stdout=logf,
                        stderr=subprocess.STDOUT,
                        check=False,
                    )
                if completed.returncode != 0:
                    self.log(
                        f"RESCORE FAIL {task.bench}: rc={completed.returncode}"
                    )
                    rc = completed.returncode

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

    @staticmethod
    def _read_jsonl(path: Path) -> list[dict]:
        rows = []
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if line.strip():
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError as exc:
                        raise ValueError(f"Invalid JSONL at {path}:{line_number}") from exc
        return rows

    def write_paired_comparisons(self):
        if not self.args.paired_baseline_labels:
            return
        from src.eval.unified_benchmark import paired_bootstrap_difference

        benches = (
            list(ALL_BENCHMARKS)
            if self.args.benchmarks == ["all"]
            else list(self.args.benchmarks)
        )
        for baseline_label in self.args.paired_baseline_labels:
            for bench in benches:
                candidate_path = self.output_dir / f"{self.args.label}_{bench}_details.jsonl"
                baseline_path = self.output_dir / f"{baseline_label}_{bench}_details.jsonl"
                if not candidate_path.exists() or not baseline_path.exists():
                    self.log(
                        f"PAIR PENDING {bench}: need both {self.args.label} and "
                        f"{baseline_label} details"
                    )
                    continue
                try:
                    paired = paired_bootstrap_difference(
                        self._read_jsonl(candidate_path),
                        self._read_jsonl(baseline_path),
                        n_resamples=10_000,
                        seed=0,
                    )
                except Exception as exc:
                    self.log(f"PAIR SKIP {bench} vs {baseline_label}: {exc}")
                    continue
                payload = {
                    "candidate_label": self.args.label,
                    "baseline_label": baseline_label,
                    "benchmark": bench,
                    "interval_scope": "evaluation_items_not_training_variance",
                    **paired,
                }
                output_path = self.output_dir / (
                    f"{self.args.label}_vs_{baseline_label}_{bench}_paired_bootstrap.json"
                )
                tmp_path = output_path.with_suffix(output_path.suffix + ".partial")
                tmp_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
                tmp_path.replace(output_path)
                self.log(
                    f"PAIR {bench} vs {baseline_label}: "
                    f"accuracy={paired['accuracy_delta_pp']:+.2f}pp, "
                    f"tokens={paired['mean_token_delta']:+.1f}"
                )

    def run(self):
        tasks = self.build_tasks()
        pending = self.prune_completed(tasks)
        if not pending:
            self.log("Nothing to do; all metrics.json exist. Use --force to re-run.")
            self.write_paired_comparisons()
            self.summarize()
            return 0
        self.log(
            f"Launching {len(pending)} benches on GPUs {self.args.gpus}: "
            f"{[t.bench for t in pending]}"
        )
        signal.signal(signal.SIGINT, lambda s, f: self.stop_event.set())
        signal.signal(signal.SIGTERM, lambda s, f: self.stop_event.set())
        self.worker_loop(pending)
        self.write_paired_comparisons()
        self.summarize()
        failed = [r for r in self.results if r.exit_code != 0]
        return 1 if failed else 0


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True, help="Local HF model path.")
    p.add_argument("--adapter", default="", help="Optional LoRA adapter dir.")
    p.add_argument("--label", required=True, help="Short run label.")
    p.add_argument("--gpus", default="0,1,2,3,4,5,6,7",
                   help="Comma-separated CUDA ids to use.")
    p.add_argument("--benchmarks", nargs="+", default=["all"],
                   help="Which benchmarks to run; 'all' uses the registered defaults.")
    p.add_argument("--num_samples_per_item", type=int, default=1)
    p.add_argument("--k_values", nargs="+", type=int, default=[1])
    p.add_argument("--pass1_do_sample", action="store_true")
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--top_k", type=int, default=20)
    p.add_argument("--min_p", type=float, default=0.0)
    p.add_argument("--repetition_penalty", type=float, default=1.0)
    p.add_argument("--eval_seed", type=int, default=0)
    p.add_argument("--fold_system_into_user", action="store_true")
    p.add_argument("--force_think_prefix", action="store_true")
    p.add_argument("--system_control", default="")
    p.add_argument("--empty_system_prompt", action="store_true")
    p.add_argument("--user_suffix", default="")
    p.add_argument("--score_topology", action="store_true")
    p.add_argument("--use_chat_template", action="store_true", default=True,
                   help="Use tokenizer chat template (default ON).")
    p.add_argument("--no_chat_template", dest="use_chat_template",
                   action="store_false", help="Disable chat template.")
    p.add_argument("--sft_style", action="store_true",
                   help="Add --sft_style (for SFT/GRPO adapters).")
    p.add_argument("--force", action="store_true",
                   help="Re-run benches even if metrics.json already exists.")
    p.add_argument("--save_solutions", action="store_true")
    p.add_argument(
        "--paired_baseline_labels",
        nargs="*",
        default=[],
        help="Existing labels in output_dir for paired item-level bootstrap deltas.",
    )
    p.add_argument("--output_dir", default="output/eval")
    p.add_argument("--log_dir", default="logs/unified")
    args = p.parse_args(argv)
    if args.repetition_penalty <= 0:
        p.error("--repetition_penalty must be greater than zero")
    if args.temperature <= 0:
        p.error("--temperature must be greater than zero")
    if not 0 < args.top_p <= 1:
        p.error("--top_p must be in (0, 1]")
    if args.top_k < 0:
        p.error("--top_k must be non-negative")
    if not 0 <= args.min_p <= 1:
        p.error("--min_p must be in [0, 1]")
    if args.eval_seed < 0:
        p.error("--eval_seed must be non-negative")
    chat_protocol_flags = (
        args.fold_system_into_user
        or args.force_think_prefix
        or args.system_control
        or args.empty_system_prompt
        or args.user_suffix
    )
    if chat_protocol_flags and not args.use_chat_template:
        p.error(
            "chat-role and model-native prompt controls require chat templates"
        )
    if args.system_control and args.empty_system_prompt:
        p.error("--system_control and --empty_system_prompt are mutually exclusive")
    args.gpus = [int(x) for x in args.gpus.split(",") if x.strip() != ""]
    return args


def main(argv=None):
    args = parse_args(argv)
    orch = Orchestrator(args)
    return orch.run()


if __name__ == "__main__":
    sys.exit(main())

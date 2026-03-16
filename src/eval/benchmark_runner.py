"""Benchmark runner that wraps ``swift eval`` for standard math benchmarks."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


SUPPORTED_BENCHMARKS: dict[str, str] = {
    "MATH": "math",
    "GSM8K": "gsm8k",
    "CMATH": "cmath",
    "GaoKao": "gaokao",
    "C-Eval-Math": "ceval_math",
}


@dataclass
class BenchmarkResult:
    """Container for a single benchmark evaluation result."""

    benchmark: str
    metrics: dict[str, Any] = field(default_factory=dict)
    output_dir: Optional[str] = None
    success: bool = True
    error: Optional[str] = None


class BenchmarkRunner:
    """Run math benchmarks via the ``swift eval`` CLI.

    Parameters
    ----------
    model_id_or_path:
        HuggingFace model id **or** a local checkpoint path.
    output_dir:
        Directory where evaluation artefacts are written.  Defaults to
        ``./eval_results``.
    extra_args:
        Additional CLI flags forwarded verbatim to ``swift eval``.
    """

    def __init__(
        self,
        model_id_or_path: str,
        output_dir: str = "./eval_results",
        extra_args: Optional[list[str]] = None,
    ) -> None:
        self.model_id_or_path = model_id_or_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.extra_args: list[str] = extra_args or []

    def _build_command(self, benchmark_key: str) -> list[str]:
        """Build the ``swift eval`` command for a given benchmark."""
        swift_dataset = SUPPORTED_BENCHMARKS[benchmark_key]
        bench_dir = self.output_dir / benchmark_key
        bench_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable, "-m", "swift", "eval",
            "--model_id_or_path", self.model_id_or_path,
            "--eval_dataset", swift_dataset,
            "--eval_output_dir", str(bench_dir),
        ]
        cmd.extend(self.extra_args)
        return cmd

    def run_benchmark(self, name: str) -> BenchmarkResult:
        """Run a single benchmark by name.

        Parameters
        ----------
        name:
            One of the keys in :data:`SUPPORTED_BENCHMARKS`
            (``MATH``, ``GSM8K``, ``CMATH``, ``GaoKao``, ``C-Eval-Math``).
        """
        if name not in SUPPORTED_BENCHMARKS:
            return BenchmarkResult(
                benchmark=name,
                success=False,
                error=f"Unsupported benchmark {name!r}. "
                       f"Choose from {sorted(SUPPORTED_BENCHMARKS)}.",
            )

        cmd = self._build_command(name)
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
            )
        except subprocess.CalledProcessError as exc:
            return BenchmarkResult(
                benchmark=name,
                success=False,
                error=f"swift eval failed (rc={exc.returncode}): {exc.stderr[:500]}",
            )

        metrics = self._parse_metrics(name)
        return BenchmarkResult(
            benchmark=name,
            metrics=metrics,
            output_dir=str(self.output_dir / name),
            success=True,
        )

    def run_all(self) -> list[BenchmarkResult]:
        """Run every supported benchmark sequentially."""
        return [self.run_benchmark(name) for name in SUPPORTED_BENCHMARKS]

    def _parse_metrics(self, benchmark_key: str) -> dict[str, Any]:
        """Try to load metrics JSON produced by ``swift eval``."""
        bench_dir = self.output_dir / benchmark_key
        for candidate in ("metrics.json", "results.json", "eval_result.json"):
            path = bench_dir / candidate
            if path.is_file():
                with open(path, encoding="utf-8") as f:
                    return json.load(f)
        return {}

    def summary(self, results: list[BenchmarkResult]) -> str:
        """Pretty-print a summary table."""
        lines: list[str] = [
            f"{'Benchmark':<15} {'Status':<10} {'Metrics'}",
            "-" * 60,
        ]
        for r in results:
            status = "OK" if r.success else "FAIL"
            metric_str = json.dumps(r.metrics, ensure_ascii=False) if r.metrics else (r.error or "—")
            lines.append(f"{r.benchmark:<15} {status:<10} {metric_str}")
        return "\n".join(lines)


def main() -> None:
    """CLI entry-point for running benchmarks."""
    parser = argparse.ArgumentParser(
        description="Run math benchmarks via swift eval",
    )
    parser.add_argument(
        "--model", required=True,
        help="HuggingFace model id or local checkpoint path",
    )
    parser.add_argument(
        "--benchmarks", nargs="*", default=list(SUPPORTED_BENCHMARKS),
        help=f"Benchmarks to run (default: all). Choices: {sorted(SUPPORTED_BENCHMARKS)}",
    )
    parser.add_argument(
        "--output_dir", default="./eval_results",
        help="Output directory (default: ./eval_results)",
    )
    args = parser.parse_args()

    runner = BenchmarkRunner(
        model_id_or_path=args.model,
        output_dir=args.output_dir,
    )

    selected = args.benchmarks
    results: list[BenchmarkResult] = []
    for bench in selected:
        print(f"▶ Running {bench} …")
        result = runner.run_benchmark(bench)
        results.append(result)
        status = "✓" if result.success else "✗"
        print(f"  {status} {bench}: {result.metrics or result.error}")

    print()
    print(runner.summary(results))


if __name__ == "__main__":
    main()

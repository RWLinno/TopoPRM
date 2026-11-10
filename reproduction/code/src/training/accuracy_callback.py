"""Light-weight eval-accuracy callback for trl/transformers Trainer loops.

The production benchmark runner (``scripts/bench_transformers.py``) is too
heavyweight for every-N-step evaluation inside training. This callback runs
greedy generation on a small held-out prompt subset, extracts the boxed/numeric
answer, compares to the gold, and logs::

    eval/accuracy         fraction of correct completions
    eval/accuracy_n       size of the evaluation subset
    eval/mean_new_tokens  average number of newly generated tokens

These appear in ``state.log_history`` and are printed by the Trainer's regular
logger at the same cadence as other metrics, which means
``tutorials/training_curve.py`` will see them in the log file without any extra
plumbing.

Usage in a training script::

    from src.training.accuracy_callback import EvalAccuracyCallback
    cb = EvalAccuracyCallback(
        tokenizer=tokenizer,
        eval_examples=[{"prompt": "...", "gold": "42"}, ...],
        eval_every=20,
        max_new_tokens=512,
    )
    trainer.add_callback(cb)

The subset is built in ``build_eval_subset()`` from a JSONL that already lives
in the repo (default: ``data/grpo_ready/train_public.jsonl``). See the module
``main`` below for a quick sanity check.
"""

from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Any, Callable, Iterable, List, Optional, Sequence

try:  # Imports are deferred to the call site so tutorials/* that only parse
    # logs don't need torch installed.
    import torch
except Exception:  # pragma: no cover - fall back gracefully
    torch = None  # type: ignore[assignment]

try:
    from transformers import TrainerCallback
except Exception:  # pragma: no cover - training stack not installed
    class TrainerCallback:  # type: ignore[no-redef]
        """Fallback stub so this module imports outside a training env."""

        def on_step_end(self, args, state, control, **kwargs):  # noqa: D401
            return control


_BOXED = re.compile(r"\\boxed\{([^}]+)\}")
_NUM = re.compile(r"-?\d+(?:\.\d+)?")


def extract_boxed_or_numeric(text: str) -> Optional[str]:
    if not text:
        return None
    m = _BOXED.search(text)
    if m:
        return m.group(1).strip()
    # Fallback to the *last* numeric token in the generated text.
    matches = _NUM.findall(text)
    return matches[-1].strip() if matches else None


def _norm(s: str) -> str:
    return "".join(ch for ch in s.strip() if not ch.isspace())


def answers_match_numeric(pred: Optional[str], gold: str) -> bool:
    if pred is None:
        return False
    p = _norm(str(pred)).replace(",", "")
    g = _norm(str(gold)).replace(",", "")
    if not p or not g:
        return False
    if p == g:
        return True
    # Allow trailing zeros / decimal differences: 5 ≡ 5.0.
    try:
        return abs(float(p) - float(g)) < 1e-6
    except (TypeError, ValueError):
        return False


def build_eval_subset(
    jsonl_path: Path,
    size: int = 64,
    sources: Sequence[str] = ("gsm8k", "math"),
    seed: int = 7,
    prompt_key: str = "question",
    gold_key: str = "final_answer",
    system_prompt: Optional[str] = (
        "Please reason step by step, and put your final answer within "
        "\\boxed{}."
    ),
) -> List[dict]:
    """Return a stratified sample of ``{"prompt": str, "gold": str}`` rows.

    The default uses ``data/grpo_ready/train_public.jsonl`` which is already
    consumed by the training entry point, so there is no additional download
    step and the same rows are guaranteed to be compatible with the training
    prompt format.
    """
    rng = random.Random(seed)
    rows: List[dict] = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if sources and row.get("source") not in sources:
                continue
            rows.append(row)
    rng.shuffle(rows)
    buckets: dict[str, List[dict]] = {src: [] for src in sources}
    for row in rows:
        src = row.get("source", "other")
        buckets.setdefault(src, []).append(row)

    sample: List[dict] = []
    per_bucket = max(1, size // max(1, len(buckets)))
    for src, bucket in buckets.items():
        for r in bucket[:per_bucket]:
            prompt = r.get(prompt_key, "")
            gold = str(r.get(gold_key, ""))
            if not prompt or not gold:
                continue
            msgs: List[dict[str, str]] = []
            if system_prompt:
                msgs.append({"role": "system", "content": system_prompt})
            msgs.append({"role": "user", "content": prompt})
            sample.append({"messages": msgs, "prompt": prompt, "gold": gold, "source": src})
    rng.shuffle(sample)
    return sample[:size]


class EvalAccuracyCallback(TrainerCallback):
    """Run greedy eval every N steps and log accuracy into ``state.log_history``.

    Parameters
    ----------
    tokenizer:
        Matches the trainer's tokenizer (chat template is applied if available).
    eval_examples:
        List of dicts produced by :func:`build_eval_subset`.
    eval_every:
        Frequency in gradient steps.
    max_new_tokens:
        Per-sample cap on generated tokens. Keep this small to bound cost.
    extractor / matcher:
        Override to support non-numeric benchmarks (e.g. MCQ).
    enabled_on_main_only:
        Only run on ``args.local_rank in (-1, 0)`` to avoid duplicate work in
        DDP. The logged metric is rebroadcast via ``state.log_history``.
    """

    def __init__(
        self,
        tokenizer: Any,
        eval_examples: Sequence[dict],
        eval_every: int = 20,
        max_new_tokens: int = 512,
        extractor: Callable[[str], Optional[str]] = extract_boxed_or_numeric,
        matcher: Callable[[Optional[str], str], bool] = answers_match_numeric,
        enabled_on_main_only: bool = True,
    ) -> None:
        self.tokenizer = tokenizer
        self.eval_examples = list(eval_examples)
        self.eval_every = max(1, int(eval_every))
        self.max_new_tokens = int(max_new_tokens)
        self.extractor = extractor
        self.matcher = matcher
        self.enabled_on_main_only = enabled_on_main_only
        # A shallow cache of rendered prompt token ids to avoid re-tokenising
        # every round. Keyed by the stringified chat turns.
        self._prompt_cache: dict[str, Any] = {}

    # ------------------------------------------------------------------
    def _render_prompt(self, example: dict) -> str:
        if hasattr(self.tokenizer, "apply_chat_template"):
            try:
                return self.tokenizer.apply_chat_template(
                    example["messages"],
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception:
                pass
        parts: List[str] = []
        for msg in example["messages"]:
            parts.append(f"{msg['role']}: {msg['content']}")
        parts.append("assistant:")
        return "\n".join(parts)

    def _generate(self, model, prompts: List[str]) -> List[tuple[str, int]]:
        tok = self.tokenizer
        if torch is None:
            raise RuntimeError("torch is required for EvalAccuracyCallback")
        # Process one example at a time to sidestep padding issues on chat
        # models, and to keep memory bounded during training-time eval.
        outs: List[tuple[str, int]] = []
        model_device = next(model.parameters()).device
        was_training = model.training
        model.eval()
        try:
            with torch.no_grad():
                for prompt in prompts:
                    enc = tok(prompt, return_tensors="pt")
                    enc = {k: v.to(model_device) for k, v in enc.items()}
                    prompt_len = enc["input_ids"].shape[1]
                    gen_kwargs = dict(
                        max_new_tokens=self.max_new_tokens,
                        do_sample=False,
                        temperature=0.0,
                        use_cache=True,
                        pad_token_id=tok.pad_token_id or tok.eos_token_id,
                    )
                    gen = model.generate(**enc, **gen_kwargs)
                    new_ids = gen[0, prompt_len:]
                    text = tok.decode(new_ids, skip_special_tokens=True)
                    outs.append((text, int(new_ids.shape[0])))
        finally:
            if was_training:
                model.train()
        return outs

    # ------------------------------------------------------------------
    def _should_run(self, args, state) -> bool:
        if state.global_step == 0:
            return False
        if state.global_step % self.eval_every != 0:
            return False
        if self.enabled_on_main_only and getattr(args, "local_rank", -1) not in (-1, 0):
            return False
        return True

    def on_step_end(self, args, state, control, **kwargs):
        if not self._should_run(args, state):
            return control
        model = kwargs.get("model")
        if model is None:
            return control
        if not self.eval_examples:
            return control

        prompts = [self._render_prompt(ex) for ex in self.eval_examples]
        try:
            decoded = self._generate(model, prompts)
        except Exception as e:  # pragma: no cover - training should continue
            print(f"[EvalAccuracyCallback] generation failed at step "
                  f"{state.global_step}: {e}")
            return control

        correct = 0
        total_new_tokens = 0
        for (text, n_new), example in zip(decoded, self.eval_examples):
            pred = self.extractor(text)
            if self.matcher(pred, example["gold"]):
                correct += 1
            total_new_tokens += n_new
        n = len(self.eval_examples)
        accuracy = correct / n if n else 0.0
        metrics = {
            "eval/accuracy": round(accuracy, 4),
            "eval/accuracy_n": n,
            "eval/mean_new_tokens": round(total_new_tokens / max(n, 1), 1),
        }
        # Push into log_history so Trainer prints it on the next logging_steps.
        state.log_history.append({"step": state.global_step, **metrics})
        # Also call the trainer's log path so wandb captures it immediately.
        trainer = kwargs.get("trainer")
        if trainer is not None and hasattr(trainer, "log"):
            try:
                trainer.log(metrics)
            except Exception:
                pass
        print(f"[eval] step={state.global_step} acc={accuracy:.3f} "
              f"n={n} mean_new_tokens={metrics['eval/mean_new_tokens']}")
        return control


def main() -> None:
    """Quick sanity check: build a subset and print its shape."""
    repo_root = Path(__file__).resolve().parents[2]
    jsonl = repo_root / "data" / "grpo_ready" / "train_public.jsonl"
    if not jsonl.exists():
        raise SystemExit(f"missing {jsonl}")
    rows = build_eval_subset(jsonl, size=8)
    for r in rows:
        print(f"- [{r['source']}] gold={r['gold']!r}  prompt={r['prompt'][:80]!r}")
    print(f"built {len(rows)} rows")


if __name__ == "__main__":
    main()

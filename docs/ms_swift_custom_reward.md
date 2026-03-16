# Custom Reward Functions in ms-swift

This guide explains how to write, register, and debug custom reward functions (ORMs) for GRPO training in [ms-swift](https://github.com/modelscope/ms-swift).

---

## ORM Interface

All custom rewards must inherit from `swift.rewards.ORM` and implement `__call__`:

```python
from swift.rewards import ORM, orms

class MyReward(ORM):
    def __call__(
        self,
        completions: list[list[dict[str, str]]],
        **kwargs,
    ) -> list[float]:
        rewards = []
        for completion in completions:
            text = completion[-1].get("content", "")
            # ... compute reward from text ...
            rewards.append(score)
        return rewards

# Register so GRPO can find it by name
orms["my_reward"] = MyReward
```

### `__call__` Signature

```python
def __call__(
    self,
    completions: list[list[dict[str, str]]],
    **kwargs,
) -> list[float]
```

| Parameter | Type | Description |
|---|---|---|
| `completions` | `list[list[dict[str, str]]]` | Batch of completions. Each completion is a list of message dicts with `"role"` and `"content"` keys. The model-generated text is typically in `completions[i][-1]["content"]`. |
| `**kwargs` | `Any` | Extra columns from the dataset record (see below). |
| **return** | `list[float]` | One reward per completion. Length **must** equal `len(completions)`. |

---

## How Dataset Extra Columns Become kwargs

In the GRPO dataset JSONL, each record has a `messages` field (the prompt) plus any extra columns:

```json
{
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."}
  ],
  "solution": "{\"score\": 8, \"conclusion\": \"部分正确\"}",
  "reference_dag": "{\"problem_id\": \"q1\", \"nodes\": [...], \"edges\": [...]}"
}
```

ms-swift passes these extra columns as keyword arguments to the ORM's `__call__`. For instance, the `TopoCompositeReward` receives:

```python
def __call__(
    self,
    completions,
    solution=None,       # ← from dataset "solution" column
    reference_dag=None,  # ← from dataset "reference_dag" column
    **kwargs,
):
    ...
```

> **Tip:** Always provide default values (e.g., `=None`) for extra-column kwargs. Not every dataset record is guaranteed to have them, and ms-swift may call the ORM during validation without supplying them.

---

## GRPO Config

To use a custom reward in GRPO training, configure two fields in the YAML:

```yaml
# Load the Python file that contains the ORM class and orms[...] registration
external_plugins: src/reward/composite_reward.py

# Reference the registered name(s)
reward_funcs:
  - topo_composite
```

Full example config (`configs/grpo_qwen3_14b.yaml`):

```yaml
model: output/sft_qwen3_14b/best_model
rlhf_type: grpo

dataset:
  - type: jsonl
    path: data/grpo_ready/train.jsonl

external_plugins: src/reward/composite_reward.py
reward_funcs:
  - topo_composite

num_generations: 8
temperature: 0.7
max_completion_length: 4096

train_type: lora
lora_rank: 64
lora_alpha: 128
deepspeed: zero2
per_device_train_batch_size: 1
gradient_accumulation_steps: 4
learning_rate: 5e-6
num_train_epochs: 1
bf16: true
output_dir: output/grpo_qwen3_14b
```

Launch:

```bash
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
swift rlhf --rlhf_type grpo --config configs/grpo_qwen3_14b.yaml
```

---

## Code Examples

### Testing a Reward Standalone

```python
#!/usr/bin/env python3
"""Quick sanity check for the composite reward."""
import sys
sys.path.insert(0, ".")  # ensure project root is importable

from src.reward.composite_reward import TopoCompositeReward

reward_fn = TopoCompositeReward()

completions = [
    [{"role": "assistant", "content": (
        "<think>\n"
        "已知三角形ABC中，AB=5，BC=3。\n"
        "由勾股定理，AC² = AB² - BC² = 25 - 9 = 16。\n"
        "故 AC = 4。\n"
        "</think>\n"
        "<answer>{\"学生得分\": 8, \"结论批改\": \"部分正确\"}</answer>"
    )}],
]

solution = '{"学生得分": 8, "结论批改": "部分正确"}'

scores = reward_fn(completions, solution=solution, reference_dag=None)
print(f"Reward: {scores}")
# Expected: a list with one float in [0, 1]
```

### Testing Individual Components

```python
from src.reward.outcome_reward import OutcomeReward
from src.reward.format_reward import FormatReward
from src.reward.topo_reward import TopoReward
from src.reward.continuity_reward import ContinuityReward
from src.reward.composite_reward import LengthReward

completions = [[{"role": "assistant", "content": "..."}]]

print("Outcome:", OutcomeReward()(completions, solution='{"score": 10}'))
print("Format: ", FormatReward()(completions))
print("Topo:   ", TopoReward()(completions))
print("Contin.:", ContinuityReward()(completions))
print("Length: ", LengthReward()(completions))
```

### Debugging with Verbose Logging

```python
import logging
logging.basicConfig(level=logging.DEBUG)

from src.reward.composite_reward import TopoCompositeReward

reward_fn = TopoCompositeReward()
scores = reward_fn(completions, solution=solution)
```

---

## Common Pitfalls

### 1. `KeyError` when accessing kwargs

**Symptom:** `KeyError: 'solution'` during GRPO training.

**Cause:** The dataset record is missing the `solution` column, or you accessed `kwargs["solution"]` instead of using a default parameter.

**Fix:** Always declare extra columns as keyword arguments with defaults:

```python
def __call__(self, completions, solution=None, **kwargs):
    if solution is None:
        return [0.0] * len(completions)
    ...
```

### 2. Return length mismatch

**Symptom:** `AssertionError` or silent shape mismatch in the GRPO trainer.

**Cause:** The returned list has a different length than `completions`.

**Fix:** Ensure you always return exactly `len(completions)` floats:

```python
def __call__(self, completions, **kwargs):
    rewards = []
    for completion in completions:
        rewards.append(self._score_one(completion))
    assert len(rewards) == len(completions)
    return rewards
```

### 3. NaN / Inf rewards

**Symptom:** Training loss becomes NaN after a few steps.

**Cause:** Division by zero or `math.log(0)` in reward computation.

**Fix:** Guard against edge cases:

```python
import math

score = numerator / denominator if denominator != 0 else 0.0
if math.isnan(score) or math.isinf(score):
    score = 0.0
```

### 4. `PYTHONPATH` not set

**Symptom:** `ModuleNotFoundError: No module named 'src'` when ms-swift tries to import the plugin.

**Cause:** The project root is not on `PYTHONPATH`.

**Fix:** Export before launching training:

```bash
cd /path/to/topoprm
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
swift rlhf --rlhf_type grpo --config configs/grpo_qwen3_14b.yaml
```

Or use the provided script which handles this automatically:

```bash
bash scripts/run_grpo.sh
```

### 5. Reward function not found

**Symptom:** `KeyError: 'topo_composite'` from ms-swift's reward lookup.

**Cause:** The `external_plugins` path is wrong or the module-level `orms[...] = ...` registration was not executed.

**Fix:** Verify:
1. `external_plugins` path is relative to the working directory (or absolute).
2. The Python file contains top-level `orms["topo_composite"] = TopoCompositeReward` statements (not hidden inside `if __name__ == "__main__"`).
3. All imports in the plugin file succeed (no import errors silently swallowed).

### 6. Extra column type mismatch

**Symptom:** `json.JSONDecodeError` when parsing `solution` or `reference_dag`.

**Cause:** The column was stored as a raw string but the reward function expects a dict.

**Fix:** Handle both types gracefully (see `OutcomeReward._parse_solution` for a reference pattern):

```python
if isinstance(solution, str):
    try:
        solution = json.loads(solution)
    except json.JSONDecodeError:
        return [0.0] * len(completions)
```

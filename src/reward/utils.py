from __future__ import annotations

import re
from typing import Any


def completion_to_text(completion: Any) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list) and completion:
        last = completion[-1]
        if isinstance(last, dict):
            return str(last.get("content", ""))
    return ""


def extract_think_block(text: str) -> str:
    m = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
    return m.group(1).strip() if m else ""


def restore_response_prefix(rendered_prompt: str, completion: str) -> str:
    """Restore a template-prefilled think opener for parsing/scoring only.

    The opener is already in the model context and must not be added to the
    generated token IDs used by GRPO or reverse KL.
    """
    if re.search(r"<think>\s*$", rendered_prompt) and not completion.lstrip().startswith("<think>"):
        return "<think>\n" + completion
    return completion

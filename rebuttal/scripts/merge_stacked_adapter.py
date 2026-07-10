#!/usr/bin/env python3
"""Merge base + SFT adapter + GRPO adapter into a standalone model.

The released grpo-topoprm-dr1-7b adapter was trained on top of the SFT adapter
(args.json: adapters=[sft_deepseek_r1_7b/final]). Evaluating the GRPO adapter
alone on the raw base model is therefore incorrect. This script reproduces the
training stack: base -> +SFT (merge) -> +GRPO (merge) -> save.

Usage:
  python rebuttal/scripts/merge_stacked_adapter.py \
    --base /Knowin/foundation/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
    --sft rebuttal/ckpts/sft-dr1-7b-final \
    --grpo rebuttal/ckpts/grpo-topoprm-dr1-7b \
    --out output/merged_topoprm_dr1_7b
"""
from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))


def _patch_namespace(adapter_dir: Path) -> Path:
    """Strip swift '.language_model.' namespace from adapter tensors if present."""
    import json
    from safetensors.torch import load_file, save_file

    tmp = Path(tempfile.mkdtemp(prefix="adapter_fix_")) / "adapter"
    shutil.copytree(adapter_dir, tmp, dirs_exist_ok=True)
    st = tmp / "adapter_model.safetensors"
    if st.exists():
        state = load_file(str(st))
        if any(".language_model." in k for k in state):
            state = {k.replace(".language_model.", "."): v for k, v in state.items()}
            save_file(state, str(st))
    return tmp


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--sft", default="")
    ap.add_argument("--grpo", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.base, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.base, torch_dtype=torch.bfloat16, device_map="cpu", trust_remote_code=True
    )

    if args.sft and Path(args.sft).is_dir():
        print(f"[merge] applying SFT: {args.sft}")
        sft = _patch_namespace(Path(args.sft))
        model = PeftModel.from_pretrained(model, str(sft))
        model = model.merge_and_unload()
        shutil.rmtree(sft.parent, ignore_errors=True)

    print(f"[merge] applying GRPO: {args.grpo}")
    grpo = _patch_namespace(Path(args.grpo))
    model = PeftModel.from_pretrained(model, str(grpo))
    model = model.merge_and_unload()
    shutil.rmtree(grpo.parent, ignore_errors=True)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out))
    tok.save_pretrained(str(out))
    print(f"[merge] saved stacked model -> {out}")


if __name__ == "__main__":
    main()

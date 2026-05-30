#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'output' / 'eval' / 'benchmark_light'
OUT.mkdir(parents=True, exist_ok=True)

# lightweight benchmark settings
EVAL_DATASETS = ['math_500', 'gsm8k']
EVAL_LIMIT = 20

MODELS = [
    {
        'name': 'grpo_main_light',
        'model': 'Qwen/Qwen3-32B',
        'adapters': 'output/grpo_main/v3-20260318-211524/checkpoint-79',
    },
    {
        'name': 'grpo_outcome_light',
        'model': 'Qwen/Qwen3-32B',
        'adapters': 'output/grpo_outcome_only/v0-20260319-171419/checkpoint-212',
    },
    {
        'name': 'baseline_qwen25_7b_light',
        'model': '${HF_MODELS_DIR:-./models}/Qwen2.5-7B-Instruct',
        'adapters': None,
    },
    {
        'name': 'baseline_llama31_8b_light',
        'model': '${HF_MODELS_DIR:-./models}/Llama-3.1-8B-Instruct',
        'adapters': None,
    },
]

summary = []
for m in MODELS:
    for ds in EVAL_DATASETS:
        run_dir = OUT / f"{m['name']}_{ds}"
        run_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            'swift', 'eval',
            '--model', m['model'],
            '--eval_dataset', ds,
            '--eval_limit', str(EVAL_LIMIT),
            '--max_new_tokens', '512',
            '--timeout', '600',
            '--eval_output_dir', str(run_dir),
        ]
        if m['adapters']:
            cmd.extend(['--adapters', m['adapters']])
        if 'Llama-3.1-8B-Instruct' in m['model']:
            cmd.extend(['--model_type', 'llama', '--template', 'llama3'])

        env = dict(os.environ)
        env['PYTHONPATH'] = str(ROOT) + (':' + env['PYTHONPATH'] if env.get('PYTHONPATH') else '')

        print('[benchmark_light] running:', ' '.join(cmd), flush=True)
        proc = subprocess.run(cmd, cwd=str(ROOT), text=True, capture_output=True, env=env)

        log_file = run_dir / 'run.log'
        log_file.write_text(proc.stdout + '\n\nSTDERR:\n' + proc.stderr, encoding='utf-8')

        status = 'ok' if proc.returncode == 0 else 'fail'
        summary.append({
            'model': m['name'],
            'dataset': ds,
            'status': status,
            'returncode': proc.returncode,
            'run_dir': str(run_dir),
        })

summary_path = OUT / 'summary.json'
summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
print('[benchmark_light] summary:', summary_path)

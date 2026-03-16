#!/bin/bash
set -euo pipefail

OUTPUT_DIR="${1:-data/benchmarks}"

echo "=== Downloading benchmark datasets ==="
mkdir -p "${OUTPUT_DIR}"

python3 -c "
from datasets import load_dataset
import os, json

output_dir = '${OUTPUT_DIR}'

benchmarks = {
    'MATH': ('hendrycks/competition_math', None),
    'GSM8K': ('openai/gsm8k', 'main'),
    'CMATH': ('weitianwen/cmath', None),
}

for name, (hf_path, subset) in benchmarks.items():
    save_dir = os.path.join(output_dir, name)
    os.makedirs(save_dir, exist_ok=True)
    print(f'Downloading {name} from {hf_path} ...')
    try:
        kwargs = {'path': hf_path}
        if subset:
            kwargs['name'] = subset
        ds = load_dataset(**kwargs)
        for split_name, split_ds in ds.items():
            out_path = os.path.join(save_dir, f'{split_name}.jsonl')
            with open(out_path, 'w', encoding='utf-8') as f:
                for item in split_ds:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
            print(f'  {split_name}: {len(split_ds)} samples -> {out_path}')
    except Exception as e:
        print(f'  WARNING: Failed to download {name}: {e}')

print('=== Benchmark download complete ===')
"

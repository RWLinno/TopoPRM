#!/usr/bin/env python3
"""Paper Stage-II entrypoint; delegates to the shared implementation."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.train_grpo_ablation import main

if __name__ == "__main__":
    main()

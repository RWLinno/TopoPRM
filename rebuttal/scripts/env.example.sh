#!/usr/bin/env bash
# Template for the rebuttal environment. Copy to env.sh and fill in secrets.
#   cp rebuttal/scripts/env.example.sh rebuttal/scripts/env.sh
# env.sh is gitignored (it holds tokens); never commit real credentials.

export TOPOPRM_ROOT=/Knowin/foundation/weilinruan/TopoPRM
export TOPOPRM_PY=/Knowin/foundation/weilinruan/env/topoprm/bin/python
export MS_SWIFT_ROOT=/Knowin/foundation/weilinruan/ms-swift

# Credentials (fill in; do NOT commit real values)
export HF_TOKEN="<your_hf_token>"
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
export WANDB_API_KEY="<your_wandb_key>"

# Optional proxy for HF/GitHub when network is poor
# export ALL_PROXY=http://<proxy-host>:80

export HF_HOME=/Knowin/foundation/weilinruan/hf_cache
export HF_HUB_ENABLE_HF_TRANSFER=0

cd "$TOPOPRM_ROOT"
export PYTHONPATH="$TOPOPRM_ROOT:${PYTHONPATH:-}"

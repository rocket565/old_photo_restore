#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

/root/miniconda3/bin/python check_models.py || true
/root/miniconda3/bin/python app.py --server-name 0.0.0.0 --server-port 6006

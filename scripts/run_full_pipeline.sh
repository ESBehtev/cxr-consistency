#!/usr/bin/env bash
set -euo pipefail

python scripts/02_prepare_task_dataset.py
python scripts/03_make_pairs.py --negatives-per-positive 2
python scripts/check_pipeline.py \
  --tokenizer-name simple \
  --max-length 256

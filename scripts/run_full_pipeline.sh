#!/usr/bin/env bash
set -euo pipefail

python scripts/01_download_and_merge.py
python scripts/02_prepare_task_dataset.py
python scripts/03_make_hard_pairs.py
python scripts/check_pipeline.py \
  --pairs-csv data/pairs/cxr_consistency_pairs_hard.csv \
  --tokenizer-name microsoft/BiomedVLP-CXR-BERT-specialized \
  --max-length 256 \
  --trust-remote-code

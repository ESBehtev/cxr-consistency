# AGENTS.md

Ultra-short operational notes for future Codex sessions in this repo.

Project: multimodal CXR image/report consistency classification, not report generation.

Goal: predict whether `(image, report)` is consistent. Label `1` means consistent, label `0` means inconsistent.

Do not break:
- patient-level split;
- `data/pairs/cxr_consistency_pairs_hard.csv`;
- CXR-BERT tokenizer/text encoder setup;
- existing MLflow logging;
- CUDA/Torch installation.

Do not run long training unless explicitly asked. Use 1-3 epoch diagnostics first.

Current main baseline:
- config: `configs/hard_pairs_convnext.yaml`
- encoder: `convnext_tiny` + `cxrbert`
- pairs: `data/pairs/cxr_consistency_pairs_hard.csv`
- negative types: `pathology_matched_report`, `distorted_negation`, `laterality_conflict`, `temporal_mismatch`, `pathology_semantic_swap`, `partial_mismatch`
- stable run: ROC-AUC about `0.918`, F1 about `0.724`, best epoch `6`

Useful comparison:
- config: `configs/vit_base_nocollapse.yaml`
- encoder: `vit_base` + `cxrbert`
- stable but weaker/slower than ConvNext: ROC-AUC about `0.871`, F1 about `0.658`

Older easier sanity baseline:
- config: `configs/best_found.yaml`
- negative types: `random_report`, `pathology_matched_report`, `distorted_negation`

Avoid unless explicitly revisiting:
- old/noisy pair types: `distorted_pathology`, old `distorted_location`, `distorted_severity`, `view_matched_report`
- old diagnostic configs and blind sweeps
- full 10+ epoch runs without epoch-1 learning signal

Debug order:
1. data integrity;
2. leakage;
3. pair quality;
4. labels/class balance;
5. tokenizer truncation;
6. logits/probs collapse;
7. model/hyperparameters.

Key commands:
```bash
python scripts/01_download_and_merge.py
python scripts/02_prepare_task_dataset.py
python scripts/03_make_hard_pairs.py
python scripts/04_train.py --config configs/hard_pairs_convnext.yaml
python scripts/04_train.py --config configs/vit_base_nocollapse.yaml
```

MLflow DBs are local SQLite files under `experiments/*/mlflow.db`.

# Final Full Runs

Prepared final full hard-pairs runs for coursework comparison. Training is not started by this README.

## Models

- `convnext_tiny_cxrbert_full`: ConvNeXt tiny + CXR-BERT unfrozen
- `deit_base_cxrbert_full`: DeiT base + CXR-BERT unfrozen
- `vit_base_cxrbert_full`: ViT base + CXR-BERT unfrozen

All configs use:

- `data/pairs/cxr_consistency_pairs_hard.csv`
- current hard negative types
- `max_train_samples: null`
- `max_valid_samples: null`
- `epochs: 20`
- early stopping on `valid_roc_auc`, patience `3`
- `use_mlflow: false`
- `save_checkpoints: false`
- AMP enabled

## Launch

```bash
tmux new -s final_full
source .venv/bin/activate
python scripts/run_final_full.py --max-parallel-jobs 3 --poll-seconds 30
```

Conservative launch:

```bash
python scripts/run_final_full.py --max-parallel-jobs 2 --poll-seconds 30
```

Dry-run check:

```bash
python scripts/run_final_full.py --dry-run --max-parallel-jobs 3 --poll-seconds 30
```

## Audit

After completion:

```bash
python scripts/audit_final_full.py
```

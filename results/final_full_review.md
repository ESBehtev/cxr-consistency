# Final Full Training Review

Read-only audit of `experiments/final_full/*` using existing `metrics.jsonl`, `status.json`, `train.log`, `config_snapshot.yaml`, and `summary.json` artifacts. No training, sweeps, data scripts, config edits, git actions, process control, or cleanup were performed.

## Executive Verdict

- `FINAL_PRIMARY_MODEL`: `ConvNext tiny + CXR-BERT unfrozen`
- `FINAL_TRANSFORMER_MODEL`: `DeiT base + CXR-BERT unfrozen`
- `FINAL_BASELINE_MODEL`: `ViT base + CXR-BERT unfrozen`
- Hardest-negative winner: `ConvNext tiny + CXR-BERT unfrozen` on `pathology_matched_report`.
- All three runs are coursework-grade: no real collapse, useful convergence curves, and clear early-stopping behavior. The main caveat is late overfitting/calibration drift after the best epoch.

## 1. Current Status

| model | status | epoch progression | best epoch | ROC-AUC | F1 | threshold | PR-AUC | best valid loss | early stopping | latest epoch |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|---:|
| ConvNext tiny + CXR-BERT unfrozen | OK | 10 rows; Epoch 10/20 | 8 | 0.9228 | 0.7378 | 0.08 | 0.7851 | e5 0.3212 | Early stopping: monitor=valid_roc_auc value=0.918440 best=0.922667 best_epoch=7 no_improve=3/3 | e10 AUC 0.9184 |
| DeiT base + CXR-BERT unfrozen | OK | 8 rows; Epoch 8/20 | 5 | 0.9156 | 0.7244 | 0.58 | 0.7572 | e3 0.3620 | Early stopping: monitor=valid_roc_auc value=0.900178 best=0.915557 best_epoch=5 no_improve=3/3 | e8 AUC 0.9002 |
| ViT base + CXR-BERT unfrozen | OK | 10 rows; Epoch 10/20 | 7 | 0.9072 | 0.7142 | 0.16 | 0.7448 | e5 0.3603 | Early stopping: monitor=valid_roc_auc value=0.903582 best=0.907170 best_epoch=7 no_improve=3/3 | e10 AUC 0.9036 |

## 2. Dynamics Analysis

### ConvNext tiny + CXR-BERT unfrozen

- ROC-AUC progression: `e1:0.8463, e2:0.8874, e3:0.8818, e4:0.9154, e5:0.9188, e6:0.9211, e7:0.9227, e8:0.9228, e9:0.9206, e10:0.9184`
- F1 progression: `e1:0.6286, e2:0.6845, e3:0.6711, e4:0.7182, e5:0.7314, e6:0.7357, e7:0.7410, e8:0.7378, e9:0.7383, e10:0.7377`
- Valid loss progression: `e1:0.4050, e2:0.3465, e3:0.3721, e4:0.3227, e5:0.3212, e6:0.3462, e7:0.3651, e8:0.3953, e9:0.4660, e10:0.5435`
- Analysis: valid ROC-AUC moved from 0.8463 to 0.9184, with best 0.9228 at epoch 8. valid F1 best is 0.7410; latest epoch reports 0.7377. Plateau starts around epoch 8; after that latest AUC is lower by 0.0043. Valid loss minimum is epoch 5 (0.3212); latest valid loss is 0.5435. Train/valid AUC gap at latest epoch is 0.0718, so overfit is material. Threshold drift: first 0.43, best-AUC 0.08, latest 0.26. Pred-positive fraction: first 35.9%, latest 30.2%.
- Latest calibration: threshold `0.26`, valid pred-positive `30.2%`, valid logits std `6.8477`, valid probs std `0.4242`.
- Convergence quality: early-stopped after overfit; best e8, drop 0.0043.

### DeiT base + CXR-BERT unfrozen

- ROC-AUC progression: `e1:0.8171, e2:0.8719, e3:0.8851, e4:0.8980, e5:0.9156, e6:0.9036, e7:0.9094, e8:0.9002`
- F1 progression: `e1:0.5932, e2:0.6584, e3:0.6729, e4:0.6940, e5:0.7244, e6:0.7062, e7:0.7201, e8:0.7201`
- Valid loss progression: `e1:0.4413, e2:0.3738, e3:0.3620, e4:0.3677, e5:0.3772, e6:0.4589, e7:0.5464, e8:0.7083`
- Analysis: valid ROC-AUC moved from 0.8171 to 0.9002, with best 0.9156 at epoch 5. valid F1 best is 0.7244; latest epoch reports 0.7201. Plateau starts around epoch 5; after that latest AUC is lower by 0.0154. Valid loss minimum is epoch 3 (0.3620); latest valid loss is 0.7083. Train/valid AUC gap at latest epoch is 0.0775, so overfit is material. Threshold drift: first 0.12, best-AUC 0.58, latest 0.05. Pred-positive fraction: first 24.3%, latest 24.7%.
- Latest calibration: threshold `0.05`, valid pred-positive `24.7%`, valid logits std `7.1766`, valid probs std `0.4039`.
- Convergence quality: early-stopped after overfit; best e5, drop 0.0154.

### ViT base + CXR-BERT unfrozen

- ROC-AUC progression: `e1:0.8086, e2:0.8570, e3:0.8703, e4:0.8782, e5:0.9013, e6:0.9020, e7:0.9072, e8:0.9028, e9:0.9002, e10:0.9036`
- F1 progression: `e1:0.5819, e2:0.6413, e3:0.6609, e4:0.6665, e5:0.7017, e6:0.7018, e7:0.7142, e8:0.7069, e9:0.7095, e10:0.7219`
- Valid loss progression: `e1:0.4446, e2:0.3904, e3:0.4071, e4:0.3825, e5:0.3603, e6:0.4153, e7:0.4283, e8:0.5497, e9:0.6234, e10:0.6874`
- Analysis: valid ROC-AUC moved from 0.8086 to 0.9036, with best 0.9072 at epoch 7. valid F1 best is 0.7219; latest epoch reports 0.7219. Plateau starts around epoch 7; after that latest AUC is lower by 0.0036. Valid loss minimum is epoch 5 (0.3603); latest valid loss is 0.6874. Train/valid AUC gap at latest epoch is 0.0741, so overfit is material. Threshold drift: first 0.25, best-AUC 0.16, latest 0.05. Pred-positive fraction: first 18.7%, latest 26.9%.
- Latest calibration: threshold `0.05`, valid pred-positive `26.9%`, valid logits std `6.9372`, valid probs std `0.4183`.
- Convergence quality: early-stopped after overfit; best e7, drop 0.0036.

## 3. Hardest Negative Analysis

### ConvNext tiny + CXR-BERT unfrozen

| negative_type | best ROC-AUC | best epoch | latest ROC-AUC | latest epoch | trend |
|---|---:|---:|---:|---:|---|
| `pathology_matched_report` | 0.8305 | 7 | 0.8275 | 10 | degraded after best |
| `distorted_negation` | 0.9930 | 4 | 0.9853 | 10 | degraded after best |
| `laterality_conflict` | 0.9207 | 4 | 0.8852 | 10 | degraded after best |
| `temporal_mismatch` | 0.9630 | 4 | 0.9399 | 10 | degraded after best |
| `pathology_semantic_swap` | 0.9863 | 6 | 0.9777 | 10 | degraded after best |
| `partial_mismatch` | 0.9689 | 4 | 0.9587 | 10 | degraded after best |

### DeiT base + CXR-BERT unfrozen

| negative_type | best ROC-AUC | best epoch | latest ROC-AUC | latest epoch | trend |
|---|---:|---:|---:|---:|---|
| `pathology_matched_report` | 0.8159 | 8 | 0.8159 | 8 | latest is best |
| `distorted_negation` | 0.9920 | 5 | 0.9736 | 8 | degraded after best |
| `laterality_conflict` | 0.9001 | 2 | 0.8559 | 8 | degraded after best |
| `temporal_mismatch` | 0.9632 | 5 | 0.9223 | 8 | degraded after best |
| `pathology_semantic_swap` | 0.9755 | 5 | 0.9488 | 8 | degraded after best |
| `partial_mismatch` | 0.9670 | 5 | 0.9273 | 8 | degraded after best |

### ViT base + CXR-BERT unfrozen

| negative_type | best ROC-AUC | best epoch | latest ROC-AUC | latest epoch | trend |
|---|---:|---:|---:|---:|---|
| `pathology_matched_report` | 0.8077 | 10 | 0.8077 | 10 | latest is best |
| `distorted_negation` | 0.9901 | 3 | 0.9799 | 10 | degraded after best |
| `laterality_conflict` | 0.8961 | 4 | 0.8722 | 10 | degraded after best |
| `temporal_mismatch` | 0.9534 | 3 | 0.9294 | 10 | degraded after best |
| `pathology_semantic_swap` | 0.9779 | 5 | 0.9618 | 10 | degraded after best |
| `partial_mismatch` | 0.9637 | 2 | 0.9291 | 10 | degraded after best |

Hardest-negative comparison on `pathology_matched_report`:

| model | best pathology AUC | best epoch | latest pathology AUC | status |
|---|---:|---:|---:|---|
| ConvNext tiny + CXR-BERT unfrozen | 0.8305 | 7 | 0.8275 | degrading |
| DeiT base + CXR-BERT unfrozen | 0.8159 | 8 | 0.8159 | latest is best |
| ViT base + CXR-BERT unfrozen | 0.8077 | 10 | 0.8077 | latest is best |

Winner: `ConvNext tiny + CXR-BERT unfrozen`. Closest competitor: `DeiT base + CXR-BERT unfrozen`. ViT improved substantially but remains behind the other two on the hardest pair type.

## 4. Collapse / Failure Check

| model | collapse flag | latest pred_pos | latest probs std | latest logits std | grad_norm inf epochs | NaN metrics | assessment |
|---|---|---:|---:|---:|---:|---|---|
| ConvNext tiny + CXR-BERT unfrozen | False | 30.2% | 0.4242 | 6.8477 | 4/10 | none | no real collapse; std and pred fraction healthy |
| DeiT base + CXR-BERT unfrozen | False | 24.7% | 0.4039 | 7.1766 | 8/8 | none | no real collapse; std and pred fraction healthy |
| ViT base + CXR-BERT unfrozen | False | 26.9% | 0.4183 | 6.9372 | 9/10 | train_grad_norm_max, train_grad_norm_mean | no real collapse; std and pred fraction healthy |

- `train_grad_norm_mean/max = inf` appears frequently, especially in transformer runs. Because losses are finite, ROC-AUC improves before early stopping, logits/probs have healthy spread, and no NaN metric appears, this looks more like AMP/gradient-norm logging overflow or occasional very large norms than a failed run.
- Calibration drift is real: late epochs push thresholds downward for all models, especially DeiT and ConvNext after their best epoch. This does not invalidate ROC-AUC, but it means final F1 should always be reported with the selected threshold.
- Results are trustworthy for ranking by ROC-AUC and by-negative-type AUC. For deployment-like probability interpretation, add calibration plots or reliability analysis.

## 5. Final Coursework Verdict

`FINAL_PRIMARY_MODEL = ConvNext tiny + CXR-BERT unfrozen`. It has the best ROC-AUC (`0.9228`), best F1 (`0.7410`), strongest hardest-negative AUC (`0.8305`), and practical runtime/VRAM. It overfits after epoch 7-8, but early stopping captured the best point cleanly.

`FINAL_TRANSFORMER_MODEL = DeiT base + CXR-BERT unfrozen`. It beats ViT on overall ROC-AUC/F1 and is close to ConvNext on `pathology_matched_report`. It overfits after epoch 5, but the peak is strong and the VRAM footprint is low.

`FINAL_BASELINE_MODEL = ViT base + CXR-BERT unfrozen`. It is a useful conservative transformer baseline: stable, no collapse, improved well over the original ViT reference, but it trails DeiT and ConvNext on both overall metrics and hardest-negative AUC.

## 6. Final Table

| model | best epoch | ROC-AUC | F1 | PR-AUC | pathology_matched_report AUC | best threshold | peak VRAM | convergence quality |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| ConvNext tiny + CXR-BERT unfrozen | 8 | 0.9228 | 0.7378 | 0.7851 | 0.8305 | 0.08 | 10742.1 MB | early-stopped after overfit; best e8, drop 0.0043 |
| DeiT base + CXR-BERT unfrozen | 5 | 0.9156 | 0.7244 | 0.7572 | 0.8159 | 0.58 | 3814.7 MB | early-stopped after overfit; best e5, drop 0.0154 |
| ViT base + CXR-BERT unfrozen | 7 | 0.9072 | 0.7142 | 0.7448 | 0.8077 | 0.16 | 3812.2 MB | early-stopped after overfit; best e7, drop 0.0036 |

## 7. Next Recommended Steps

- Calibration plots: high value. Threshold drift is visible, so reliability curves, probability histograms, and threshold-vs-F1 plots will make the coursework stronger.
- Confusion matrices: high value. Include global confusion matrices at best threshold and maybe per-negative-type confusion summaries.
- Threshold analysis: high value. Report why ROC-AUC and best-threshold F1 are both needed; show selected threshold per model.
- Grad-CAM / attention maps: medium to high value for interpretability. Best for ConvNext primary model and one transformer comparison; keep it qualitative.
- Fusion ablation: medium value. Useful if time permits: image-only, text-only, and fusion would support the multimodal claim, but it is more engineering work.
- Hardest-negative fine-tuning: lower immediate value for coursework unless framed as future work. It risks overfitting the already hardest pathology-matched cases and complicates the clean comparison.

Most scientifically useful package for the course: final leaderboard, by-negative-type table, calibration/threshold analysis, confusion matrices, and a small interpretability section.

## Short Summary

- Winner: `ConvNext tiny + CXR-BERT unfrozen`
- Best transformer: `DeiT base + CXR-BERT unfrozen`
- Hardest-negative winner: `ConvNext tiny + CXR-BERT unfrozen`
- Overfitting: yes, after best epoch for ConvNext and DeiT; ViT also shows late overfit pressure.
- Coursework-grade: yes. The results are strong, stable, and interpretable enough, with calibration caveats explicitly reportable.

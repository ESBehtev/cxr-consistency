# Final Visual Encoder Selection

This report selects visual encoders for final coursework runs from existing local sweep/finalist/text-sweep artifacts only. No training, data generation, pair generation, sweeps, git operations, or config edits were run for this analysis.

## Decision

- `FINAL_PRIMARY_VISUAL_ENCODER`: `convnext_tiny`
- `FINAL_VIT_ENCODER`: `vit_base`
- `FINAL_DEIT_ENCODER`: `deit_base`

Recommended final comparison:

1. `convnext_tiny` + CXR-BERT unfrozen
2. `deit_base` + CXR-BERT unfrozen
3. `vit_base` + CXR-BERT unfrozen

## Evidence By Encoder

### convnext_tiny

- `convnext_tiny_cosine_text_cxrbert_unfrozen_compact`: ROC-AUC 0.8644, F1 0.6490, threshold 0.42, epoch 3, VRAM 10.5GB, hardest pathology_matched_report:0.6185
- `convnext_tiny_cosine_final`: ROC-AUC 0.8505, F1 0.6328, threshold 0.29, epoch 1, VRAM 0.0GB, hardest pathology_matched_report:0.5799
- `convnext_tiny_cosine`: ROC-AUC 0.8448, F1 0.6355, threshold 0.12, epoch 5, VRAM 10.5GB, hardest pathology_matched_report:0.5741
- `convnext_tiny_dropout`: ROC-AUC 0.8419, F1 0.6383, threshold 0.42, epoch 5, VRAM 10.5GB, hardest pathology_matched_report:0.5816

### convnext_small

- `convnext_small_regularized`: ROC-AUC 0.8380, F1 0.6362, threshold 0.40, epoch 3, VRAM 7.4GB, hardest pathology_matched_report:0.5459
- `convnext_small_baseline`: ROC-AUC 0.8225, F1 0.6122, threshold 0.25, epoch 4, VRAM 9.0GB, hardest pathology_matched_report:0.5656

### convnext_base

- `convnext_base_baseline`: ROC-AUC 0.8387, F1 0.6341, threshold 0.31, epoch 6, VRAM 6.5GB, hardest pathology_matched_report:0.5599
- `convnext_base_low_lr`: ROC-AUC 0.8335, F1 0.6246, threshold 0.18, epoch 5, VRAM 6.5GB, hardest pathology_matched_report:0.5464

### vit_small

- `vit_small_regularized`: ROC-AUC 0.8355, F1 0.6293, threshold 0.35, epoch 4, VRAM 3.3GB, hardest pathology_matched_report:0.5511
- `vit_small_nocollapse`: ROC-AUC 0.8351, F1 0.6307, threshold 0.16, epoch 3, VRAM 3.3GB, hardest pathology_matched_report:0.5421

### vit_base

- `vit_base_long_warmup_text_cxrbert_unfrozen_compact`: ROC-AUC 0.8420, F1 0.6216, threshold 0.15, epoch 3, VRAM 3.7GB, hardest pathology_matched_report:0.5603
- `vit_base_long_warmup`: ROC-AUC 0.8377, F1 0.6337, threshold 0.17, epoch 4, VRAM 3.7GB, hardest pathology_matched_report:0.5662
- `vit_base_low_lr`: ROC-AUC 0.8333, F1 0.6253, threshold 0.22, epoch 4, VRAM 3.7GB, hardest pathology_matched_report:0.5565
- `vit_base_nocollapse`: ROC-AUC 0.8290, F1 0.6289, threshold 0.13, epoch 3, VRAM 3.7GB, hardest pathology_matched_report:0.5392

### deit_small

- `deit_small_nocollapse`: ROC-AUC 0.8336, F1 0.6260, threshold 0.22, epoch 4, VRAM 3.3GB, hardest pathology_matched_report:0.5681

### deit_base

- `deit_base_nocollapse_text_cxrbert_unfrozen_compact`: ROC-AUC 0.8603, F1 0.6469, threshold 0.26, epoch 3, VRAM 3.7GB, hardest pathology_matched_report:0.6218
- `deit_base_nocollapse`: ROC-AUC 0.8385, F1 0.6368, threshold 0.10, epoch 5, VRAM 3.7GB, hardest pathology_matched_report:0.5620

## Engineering Assessment

### ConvNet Family

`convnext_tiny` is the best ConvNet choice. It has the strongest overall observed result (`convnext_tiny_cosine_text_cxrbert_unfrozen_compact`, ROC-AUC 0.8644, F1 0.6490), repeated stable CXR-BERT runs around ROC-AUC 0.84+, no collapse, and a strong hardest-type result on `pathology_matched_report` (0.6185 in the compact text run). It is also the known stable baseline family from the recovery notes.

`convnext_small` is not worth promoting to final: it is competitive but inconsistent (`0.8380` regularized vs `0.8225` baseline), and does not beat tiny despite greater capacity. `convnext_base` is also not compelling: it reaches `0.8387`/`0.8335`, roughly below tiny while being more complex. Both look like capacity scaling did not pay off in this setup.

### ViT Family

`vit_base` is the selected ViT. Its long-warmup/CXR-BERT compact result reaches ROC-AUC 0.8420, and the sweep long-warmup run is close at 0.8377. It is stable under the no-collapse recipe. Calibration thresholds are low (`0.15-0.17`), so threshold tuning must be reported, but it is the best ViT representative.

`vit_small` is stable and efficient, but its best rows (`0.8355`, `0.8351`) trail `vit_base`. It is a good fallback if runtime is tight, not the main final ViT.

### DeiT Family

`deit_base` is the selected DeiT and the strongest transformer-style visual encoder. Its compact CXR-BERT result reaches ROC-AUC 0.8603 and F1 0.6469, very close to the best ConvNeXt, with low peak VRAM in the recorded runs. It also has the best `pathology_matched_report` AUC among top compact models (0.6218).

`deit_small` is stable but weaker (ROC-AUC 0.8336). It is not the final DeiT choice.

## Hardest Negative Type

`pathology_matched_report` is consistently the hardest negative type. Top compact models:

- ConvNeXt tiny + CXR-BERT unfrozen: `pathology_matched_report:0.6185`
- DeiT base + CXR-BERT unfrozen: `pathology_matched_report:0.6218`
- ViT base + CXR-BERT unfrozen: `pathology_matched_report:0.5603`

This supports choosing ConvNeXt tiny and DeiT base: they are not only high-AUC overall, they also handle the most clinically confusable negative type better than ViT base.

## Stability / Collapse / Calibration

- No selected visual encoder collapsed with CXR-BERT unfrozen.
- The only exported collapsed model is the `simple` text baseline, not a visual encoder failure.
- `convnext_tiny` is the most consistent across repeated runs and recipes.
- `deit_base` is the best balanced transformer candidate: strong AUC/F1, low recorded VRAM, strong hardest-type AUC.
- `vit_base` is stable only under the conservative no-collapse recipe; keep the long-warmup setup.
- Thresholds vary substantially (`0.42` ConvNeXt compact, `0.26` DeiT compact, `0.15` ViT compact), so final reporting should use best-threshold F1 and report thresholds explicitly.

## Undertrained / Overcomplex / Discard

Likely undertrained or incomplete:

- Finalist rows with best epoch `1` or missing metrics should not drive selection; they are useful as partial evidence only.
- Compact text runs with best epoch `3` are selection evidence, not final metrics.

Overcomplex without payoff:

- `convnext_small`: no clear gain over `convnext_tiny`.
- `convnext_base`: below tiny despite higher capacity.
- `vit_base_small_batch`: slow and weaker.

Discard for final visual comparison:

- `convnext_small`, `convnext_base` unless extra ablation time exists.
- `vit_small` as final main ViT, though it can be a runtime fallback.
- `deit_small` because `deit_base` is clearly stronger.

## FINAL_FULL_RUNS

Recommended coursework-grade final runs:

| role | image encoder | text encoder | config basis | epochs | patience | batch size | save_checkpoints |
|---|---|---|---|---:|---:|---:|---|
| primary | `convnext_tiny` | CXR-BERT unfrozen | `convnext_tiny_cosine_text_cxrbert_unfrozen_compact` / `convnext_tiny_cosine` | 8-10 | 3 | 48 | false |
| transformer comparison | `deit_base` | CXR-BERT unfrozen | `deit_base_nocollapse_text_cxrbert_unfrozen_compact` | 8-10 | 3 | 8 | false |
| ViT comparison | `vit_base` | CXR-BERT unfrozen | `vit_base_long_warmup_text_cxrbert_unfrozen_compact` | 8-10 | 3 | 8 | false |

Use `save_checkpoints: false` unless checkpoint IO/storage has been deliberately fixed; metrics artifacts are enough for coursework reporting and avoid repeating the checkpoint failure.

## Dataset Budget Recommendation

Use `max_train_samples: 60000`, `max_valid_samples: 12000` for final coursework runs. Full dataset training was already observed to be too slow for iteration, and 60k/12k is a pragmatic, controlled, comparable budget across encoders. It is large enough for meaningful comparison while staying feasible on the A6000.

## Parallelism Recommendation

- `max_parallel_jobs: 3` for the three final runs together on A6000.
- If memory or IO becomes unstable, run `convnext_tiny` + one transformer at a time with `max_parallel_jobs: 2`.

## Final Selection Summary

- `FINAL_PRIMARY_VISUAL_ENCODER = convnext_tiny`
- `FINAL_VIT_ENCODER = vit_base`
- `FINAL_DEIT_ENCODER = deit_base`

These choices balance ROC-AUC/F1, no-collapse stability, hardest-negative behavior, calibration transparency, runtime/VRAM practicality, and consistency across available runs.

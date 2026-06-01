# RECOVERY.md

Recovery playbook for restoring the current working CXR consistency project from a clean Ubuntu server plus a git clone.

## 1. Server bootstrap

Install system packages:

```bash
sudo apt update
sudo apt install -y git tmux htop unzip zip curl wget build-essential python3 python3-venv python3-dev libgl1 libglib2.0-0 jq
```

Why: Python build tools, image loading libraries, tmux sessions, and basic monitoring.

Check GPU and driver:

```bash
nvidia-smi
watch -n 2 nvidia-smi
htop
```

Expected: NVIDIA GPU is visible. Do not reinstall Torch/CUDA if this already works.

Set git identity if needed:

```bash
git config --global user.name "Your Name"
git config --global user.email "you@example.com"
```

## 2. Repository setup

Clone and enter the repo:

```bash
git clone <REPO_URL> cxr-consistency
cd cxr-consistency
git checkout server-recovery-3090
```

If that branch is unavailable, use the branch that contains `configs/hard_pairs_convnext.yaml` and `configs/vit_base_nocollapse.yaml`.

Create the environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
pip install -e .
```

Headless OpenCV note: `requirements.txt` currently has `opencv-python`. If OpenCV fails on a headless server:

```bash
pip uninstall -y opencv-python
pip install opencv-python-headless
```

Check imports and CUDA:

```bash
python - <<'PY'
import torch, timm, transformers, mlflow, pandas
print('torch', torch.__version__)
print('cuda_available', torch.cuda.is_available())
if torch.cuda.is_available():
    print('gpu', torch.cuda.get_device_name(0))
PY
```

Known pitfalls:
- Do not reinstall Torch/CUDA unless CUDA is actually broken.
- CXR-BERT needs `trust_remote_code: true`.
- First HuggingFace model/tokenizer load needs network unless cached.
- Use `local_files_only: true` only after cache warmup is verified.

## 3. Dataset setup

Data sources used by `scripts/01_download_and_merge.py`:
- Kaggle: `simhadrisadaram/mimic-cxr-dataset`
- HuggingFace: `erjui/csrrg_findings`

Configure Kaggle:

```bash
mkdir -p ~/.kaggle
nano ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
kaggle datasets list -s mimic-cxr
```

HuggingFace login is usually not required, but can help with rate/auth issues:

```bash
huggingface-cli login
```

Download raw inputs:

```bash
python scripts/01_download_and_merge.py
```

Expected outputs:
- `data/raw/kaggle_mimic/mimic_cxr_aug_train.csv`
- `data/raw/kaggle_mimic/mimic_cxr_aug_validate.csv`
- `data/raw/kaggle_mimic/official_data_iccv_final/files/**/*.jpg`
- `data/processed/hf_findings.csv`

Current reference checks:
- `data/raw`: about `15G`
- JPG files under `data`: `220210`
- CSV files under `data`: `6`

Verify extraction:

```bash
find data -type f -iname '*.jpg' | wc -l
find data -type f -iname '*.csv' | wc -l
du -sh data/raw data/processed data/pairs
```

## 4. Clean dataset pipeline

Prepare clean image/report rows:

```bash
python scripts/02_prepare_task_dataset.py
```

Expected output: `data/processed/cxr_reports_clean.csv`.

Current reference stats:
- rows: `135481`
- columns: `subject_id`, `study_id`, `image_name`, `view`, `report`, `source_split`, `image_path`, `split`
- split: train `108355`, valid `13768`, test `13358`
- broken image paths: `0`

Quick check:

```bash
python - <<'PY'
import pandas as pd
from pathlib import Path
df = pd.read_csv('data/processed/cxr_reports_clean.csv')
print(len(df))
print(df['split'].value_counts().to_dict())
print((~df['image_path'].map(lambda p: Path(p).exists())).sum())
PY
```

## 5. Pair generation pipeline

Old/easier pairs:

```bash
python scripts/03_make_pairs.py
```

Output: `data/pairs/cxr_consistency_pairs.csv`.

Current reference stats:
- rows: `406443`
- labels: `{0: 270962, 1: 135481}`
- includes easier/noisier types: `random_report`, `view_matched_report`, old distortions

Main hard pairs:

```bash
python scripts/03_make_hard_pairs.py
```

Output: `data/pairs/cxr_consistency_pairs_hard.csv`.

Current hard pair stats:
- rows: `676730`
- labels: `{0: 541249, 1: 135481}`
- split: train `541648`, valid `68683`, test `66399`
- broken image paths: `0`

Hard pair distribution:
- `positive`: `135481`
- `random_report`: `135469`
- `pathology_matched_report`: `123095`
- `distorted_negation`: `111829`
- `laterality_conflict`: `50166`
- `partial_mismatch`: `43994`
- `pathology_semantic_swap`: `41054`
- `temporal_mismatch`: `35642`

Main hard training configs filter out `random_report` and use:
- `pathology_matched_report`
- `distorted_negation`
- `laterality_conflict`
- `temporal_mismatch`
- `pathology_semantic_swap`
- `partial_mismatch`

Avoid for final hard baseline unless explicitly debugging:
- old `distorted_pathology`
- old/current noisy `distorted_location`
- `distorted_severity`
- `view_matched_report`

Leakage/integrity check:

```bash
python - <<'PY'
import pandas as pd
from pathlib import Path
p = 'data/pairs/cxr_consistency_pairs_hard.csv'
df = pd.read_csv(p)
print('rows', len(df))
print('labels', df['label'].value_counts().to_dict())
print('types', df['negative_type'].value_counts().to_dict())
print('splits', df['split'].value_counts().to_dict())
print('missing_paths', (~df['image_path'].map(lambda x: Path(x).exists())).sum())
for col in ['subject_id', 'study_id', 'image_path']:
    sets = {s: set(df.loc[df['split'] == s, col].astype(str)) for s in ['train', 'valid', 'test']}
    print(col, len(sets['train'] & sets['valid']), len(sets['train'] & sets['test']), len(sets['valid'] & sets['test']))
neg = df[df['label'] == 0]
print('same_subject_negatives', (neg['subject_id'].astype(str) == neg['paired_report_subject_id'].astype(str)).sum())
print('same_study_negatives', (neg['study_id'].astype(str) == neg['paired_report_study_id'].astype(str)).sum())
print('duplicate_pairs', df.duplicated(['image_path', 'report', 'label']).sum())
print('empty_reports', df['report'].isna().sum() + df['report'].astype(str).str.strip().eq('').sum())
PY
```

Hard pair logic is in `scripts/03_make_hard_pairs.py`.

Important guards:
- laterality swaps only in pathology/anatomy sentences;
- device/tube/line/procedure/admin text is excluded;
- semantic swaps avoid duplicate target pathology in the same sentence;
- temporal swaps require pathology context;
- partial mismatch changes one finding in multi-finding reports.

## 6. Stable training configs

Main hard ConvNext baseline:

```bash
python scripts/04_train.py --config configs/hard_pairs_convnext.yaml
```

Config summary:
- `convnext_tiny` + `cxrbert`
- hard pairs CSV
- image/text encoders unfrozen
- `lr=3e-5`, `batch_size=48`, `max_length=256`
- `amp=true`, `grad_clip_norm=1.0`
- `epochs=10`
- early stopping on `valid_roc_auc`, patience `2`, min_delta `0.001`
- checkpoints disabled

Reference result:
- best valid ROC-AUC about `0.918`
- best valid F1 about `0.724`
- best threshold about `0.23`
- best epoch `6`
- early stopped at epoch `8`
- hardest type: `pathology_matched_report`, AUC about `0.816`
- epoch time about `9.9 min`
- peak train VRAM about `10.8GB`

Older easier sanity baseline:

```bash
python scripts/04_train.py --config configs/best_found.yaml
```

Use only as easier sanity baseline:
- `random_report`, `pathology_matched_report`, `distorted_negation`
- `convnext_tiny` + `cxrbert`
- `epochs=3`

ViT comparison:

```bash
python scripts/04_train.py --config configs/vit_base_nocollapse.yaml
```

Config summary:
- `vit_base` + `cxrbert`
- hard pairs CSV
- `lr=1e-5`, `batch_size=8`, `max_length=128`
- cosine scheduler, warmup `0.1`
- image/text unfrozen
- `epochs=3`

Reference result:
- best valid ROC-AUC about `0.871`
- best valid F1 about `0.658`
- best threshold about `0.31`
- best epoch `3`
- no collapse
- slower than ConvNext: about `16.5-17.1 min/epoch`
- peak train VRAM about `3.8GB`

Coursework recommendation:
- main model: ConvNext hard baseline;
- comparison section: ViT base no-collapse;
- do not present easier `best_found` as final hard result.

## 7. MLflow

Local SQLite MLflow DBs:
- `experiments/hard_pairs_convnext/mlflow.db`
- `experiments/vit_base_nocollapse/mlflow.db`
- `experiments/best_found/mlflow.db`

Start UI:

```bash
mlflow ui --backend-store-uri sqlite:///experiments/hard_pairs_convnext/mlflow.db --host 0.0.0.0 --port 5000
```

List recent runs:

```bash
python - <<'PY'
import sqlite3
db = 'experiments/hard_pairs_convnext/mlflow.db'
con = sqlite3.connect(db)
for row in con.execute('select run_uuid, status, start_time, end_time from runs order by start_time desc limit 10'):
    print(row)
PY
```

Training logs include train/valid loss, ROC-AUC, PR-AUC, F1, best F1, best threshold, logits/probs stats, prediction positive fraction, by-negative-type metrics, and early stopping values.

## 8. tmux workflow

Create session:

```bash
tmux new -s cxr
```

Inside tmux:

```bash
source .venv/bin/activate
python scripts/04_train.py --config configs/hard_pairs_convnext.yaml
```

Detach: press `Ctrl-b d`.

Reattach:

```bash
tmux attach -t cxr
```

List sessions:

```bash
tmux ls
```

Stop a run cleanly:
- press `Ctrl-C` inside tmux first;
- if detached/lost:

```bash
pgrep -af scripts/04_train.py
kill -TERM <PID>
```

Kill session only after stopping the process:

```bash
tmux kill-session -t cxr
```

## 9. Common failure recovery

CUDA unavailable:
- run `nvidia-smi`;
- check `python -c "import torch; print(torch.cuda.is_available())"`;
- do not reinstall drivers unless GPU is invisible system-wide.

CUDA OOM:
- ConvNext: reduce batch `48 -> 32`;
- ViT base: keep batch `8`;
- reduce `num_workers` if dataloader memory is unstable;
- keep AMP enabled unless diagnosing numeric instability.

HF tokenizer/model errors:
- keep `trust_remote_code: true`;
- keep `local_files_only: false` on a fresh server;
- verify network/cache before changing configs.

Kaggle auth failure:
- verify `~/.kaggle/kaggle.json`;
- run `chmod 600 ~/.kaggle/kaggle.json`;
- test `kaggle datasets list -s mimic-cxr`.

Missing image paths:
- rerun `scripts/01_download_and_merge.py`;
- verify `data/raw/kaggle_mimic/official_data_iccv_final/files`;
- rerun `scripts/02_prepare_task_dataset.py`;
- do not hand-edit CSV paths.

Full disk:

```bash
df -h
du -sh data experiments ~/.cache/huggingface ~/.cache/torch
```

Safe cleanup targets:
- old unused `experiments/*` artifacts;
- downloaded zip after successful extraction;
- temporary caches.

Do not delete:
- `data/raw/kaggle_mimic/official_data_iccv_final/files`;
- `data/processed/cxr_reports_clean.csv`;
- `data/pairs/cxr_consistency_pairs_hard.csv`;
- final MLflow DBs unless backed up.

Constant probability collapse:
- inspect epoch-1 ROC-AUC, logits std, probs std, and `pred_positive_fraction`;
- if AUC `<=0.55` with near-zero probs/logits std, stop early;
- for ConvNext, use `configs/hard_pairs_convnext.yaml`;
- for ViT, use `configs/vit_base_nocollapse.yaml`;
- do not change split or pair generator as the first response.

Overfitting:
- ConvNext hard baseline improved until about epoch `6`, then validation stopped improving;
- rely on early stopping;
- report best epoch, not final epoch, when they differ.

Server reset:
1. SSH back in.
2. `cd cxr-consistency`.
3. `tmux ls`.
4. Reattach if session survived.
5. If not, inspect MLflow DB and `experiments/`.
6. Resume only if the previous run did not finish.

## 10. Final recommended experiments

Run first after recovery:

```bash
python scripts/04_train.py --config configs/hard_pairs_convnext.yaml
```

Then run encoder comparison:

```bash
python scripts/04_train.py --config configs/vit_base_nocollapse.yaml
```

Do not spend time on:
- blind large sweeps;
- 12+ epoch runs without early signal;
- noisy old hard negatives;
- BiomedCLIP unless HF/timm integration is intentionally revisited;
- changing data split to improve metrics.

For any new diagnostic experiment:
- keep the same hard pairs CSV;
- keep the same negative types;
- run 1 epoch first;
- stop if ROC-AUC is random and logits/probs are constant;
- compare to ConvNext before changing data.

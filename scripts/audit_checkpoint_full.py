from __future__ import annotations

from pathlib import Path
import argparse
import csv
import json
import math

try:
    import yaml
except ModuleNotFoundError:
    yaml = None

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_ROOT = ROOT / "experiments" / "checkpoint_full"
DEFAULT_OUT_DIR = ROOT / "results" / "checkpoint_full"
FIELDS = [
    "run_name", "status", "batch_size", "best_epoch", "best_roc_auc", "best_pr_auc",
    "best_f1", "best_threshold", "best_pathology_matched_report_roc_auc",
    "latest_epoch", "latest_valid_roc_auc", "peak_vram_mb", "train_time_min",
    "has_best_model", "has_last_model", "best_model_size_mb", "last_model_size_mb",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Read-only audit for checkpoint_full runs.")
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--include-running", action="store_true")
    return parser.parse_args()


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open() as f:
        return json.load(f)


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def read_yaml(path: Path) -> dict:
    if not path.exists() or yaml is None:
        return {}
    with path.open() as f:
        return yaml.safe_load(f) or {}


def as_float(value, default=0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def size_mb(path: Path) -> float:
    return path.stat().st_size / 1024 / 1024 if path.exists() else 0.0


def best_row(rows: list[dict]) -> dict:
    valid = [row for row in rows if "valid_roc_auc" in row]
    if valid:
        return max(valid, key=lambda row: as_float(row.get("valid_roc_auc"), -math.inf))
    return rows[-1] if rows else {}


def discover(root: Path, include_running: bool) -> list[Path]:
    if not root.exists():
        return []
    runs = [p for p in root.iterdir() if p.is_dir()]
    if include_running:
        return sorted(runs)
    return sorted(p for p in runs if (p / "summary.json").exists())


def collect(run_dir: Path) -> dict:
    cfg = read_yaml(run_dir / "config_snapshot.yaml")
    summary = read_json(run_dir / "summary.json")
    status = read_json(run_dir / "status.json")
    rows = read_jsonl(run_dir / "metrics.jsonl")
    best = summary.get("best_checkpoint_metrics") or best_row(rows)
    latest = rows[-1] if rows else {}
    best_model = run_dir / "best_model.pt"
    last_model = run_dir / "last_model.pt"
    return {
        "run_name": str(summary.get("run_name") or cfg.get("run_name") or status.get("run_name") or run_dir.name),
        "status": str(summary.get("status") or status.get("status") or "RUNNING"),
        "batch_size": int(cfg.get("batch_size") or status.get("batch_size") or 0),
        "best_epoch": summary.get("best_epoch") or summary.get("early_stopping_best_epoch") or best.get("epoch") or "",
        "best_roc_auc": as_float(summary.get("best_metric"), as_float(best.get("roc_auc", best.get("valid_roc_auc")))),
        "best_pr_auc": as_float(best.get("pr_auc", best.get("valid_pr_auc"))),
        "best_f1": as_float(best.get("best_f1", best.get("valid_best_f1", best.get("f1", best.get("valid_f1"))))),
        "best_threshold": as_float(summary.get("best_checkpoint_threshold"), as_float(best.get("best_threshold", best.get("valid_best_threshold")))),
        "best_pathology_matched_report_roc_auc": as_float(best.get("by_type_pathology_matched_report_roc_auc", best.get("valid_by_type_pathology_matched_report_roc_auc"))),
        "latest_epoch": latest.get("epoch") or "",
        "latest_valid_roc_auc": as_float(latest.get("valid_roc_auc")),
        "peak_vram_mb": as_float(summary.get("peak_vram_mb"), as_float(status.get("peak_vram_mb"))),
        "train_time_min": as_float(summary.get("runtime_min"), as_float(status.get("runtime_min"))),
        "has_best_model": best_model.exists(),
        "has_last_model": last_model.exists(),
        "best_model_size_mb": round(size_mb(best_model), 2),
        "last_model_size_mb": round(size_mb(last_model), 2),
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


def main() -> None:
    args = parse_args()
    rows = [collect(run_dir) for run_dir in discover(args.results_root, args.include_running)]
    rows.sort(key=lambda row: (as_float(row["best_roc_auc"]), as_float(row["best_f1"])), reverse=True)
    if not rows:
        print(f"No checkpoint_full runs found under {args.results_root}")
        return
    write_csv(args.out_dir / "runs_summary.csv", rows)
    write_json(args.out_dir / "runs_summary.json", rows)
    print(f"Wrote checkpoint_full audit to {args.out_dir}")
    print("run_name                                      status     bs  epoch auc     pr_auc  f1      thr   path_auc latest latest_auc vram  min   best last sizes_mb")
    for row in rows:
        print(
            f"{row['run_name'][:45]:45} {row['status'][:10]:10} {row['batch_size']:>3} {str(row['best_epoch']):>5} "
            f"{as_float(row['best_roc_auc']):.4f} {as_float(row['best_pr_auc']):.4f} {as_float(row['best_f1']):.4f} {as_float(row['best_threshold']):.2f} "
            f"{as_float(row['best_pathology_matched_report_roc_auc']):.4f} {str(row['latest_epoch']):>6} {as_float(row['latest_valid_roc_auc']):.4f} "
            f"{as_float(row['peak_vram_mb']):5.0f} {as_float(row['train_time_min']):6.1f} {str(row['has_best_model']).lower():4} {str(row['has_last_model']).lower():4} "
            f"{as_float(row['best_model_size_mb']):.1f}/{as_float(row['last_model_size_mb']):.1f}"
        )


if __name__ == "__main__":
    main()

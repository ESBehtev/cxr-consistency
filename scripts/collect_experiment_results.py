from __future__ import annotations

from pathlib import Path
import argparse
import csv
import json
import math
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_ROOT = ROOT / "experiments" / "sweeps"
DEFAULT_OUT_DIR = ROOT / "results" / "leaderboard"
SUMMARY_FIELDS = [
    "run_name",
    "status",
    "family",
    "image_encoder_name",
    "text_encoder_name",
    "tokenizer_name",
    "pairs_csv",
    "include_negative_types",
    "epochs_completed",
    "best_epoch",
    "best_valid_roc_auc",
    "best_valid_f1",
    "best_valid_threshold",
    "best_valid_pr_auc",
    "best_valid_loss",
    "peak_vram_mb",
    "runtime_min",
    "collapse_detected",
    "batch_size",
    "learning_rate",
    "weight_decay",
    "dropout",
    "scheduler_type",
    "warmup_ratio",
    "max_train_samples",
    "max_valid_samples",
    "rank_roc_auc",
    "rank_f1",
    "run_dir",
]
BY_TYPE_FIELDS = [
    "run_name",
    "family",
    "image_encoder_name",
    "best_epoch",
    "negative_type",
    "roc_auc",
    "f1",
    "pr_auc",
    "accuracy",
    "count_positive",
    "count_negative",
    "run_dir",
]
BEST_BY_FAMILY_FIELDS = SUMMARY_FIELDS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect local experiment summaries into leaderboard files.")
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--sort-by", choices=["roc_auc", "f1", "pr_auc", "loss", "runtime", "vram"], default="roc_auc")
    parser.add_argument("--top-k", type=int, default=0, help="Limit printed preview; 0 means no limit for saved files.")
    parser.add_argument("--include-running", action="store_true", help="Include runs without summary.json using status/metrics so far.")
    parser.add_argument("--dry-run", action="store_true", help="Discover runs and print counts without writing output files.")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open() as f:
        return json.load(f)


def read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open() as f:
        return yaml.safe_load(f) or {}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open() as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL in {path}:{line_no}: {exc}") from exc
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: csv_value(row.get(key)) for key in fields})


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


def csv_value(value: Any) -> Any:
    if isinstance(value, (list, dict)):
        return json.dumps(value, sort_keys=True)
    if value is None:
        return ""
    return value


def to_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def infer_family(config: dict[str, Any], run_dir: Path) -> str:
    family = config.get("run_family")
    if family:
        return str(family)
    encoder = str(config.get("image_encoder_name") or config.get("model_type") or "").lower()
    if encoder.startswith("convnext"):
        return "convnext"
    if encoder.startswith("deit"):
        return "deit"
    if encoder.startswith("vit"):
        return "vit"
    name = run_dir.name.lower()
    if name.startswith("convnext"):
        return "convnext"
    if name.startswith("deit"):
        return "deit"
    if name.startswith("vit"):
        return "vit"
    return "unknown"


def discover_run_dirs(results_root: Path, include_running: bool) -> list[Path]:
    if not results_root.exists():
        return []
    marker_names = {"summary.json", "metrics.jsonl", "config_snapshot.yaml"}
    run_dirs = {path.parent for path in results_root.rglob("*") if path.is_file() and path.name in marker_names}
    if not include_running:
        run_dirs = {run_dir for run_dir in run_dirs if (run_dir / "summary.json").exists()}
    return sorted(run_dirs)


def best_row(rows: list[dict[str, Any]], metric: str, mode: str = "max") -> tuple[int | None, dict[str, Any]]:
    if not rows:
        return None, {}
    key = f"valid_{metric}"
    candidates = [row for row in rows if key in row]
    if not candidates:
        return None, rows[-1]
    if mode == "min":
        row = min(candidates, key=lambda item: to_float(item.get(key), math.inf))
    else:
        row = max(candidates, key=lambda item: to_float(item.get(key), -math.inf))
    return int(row.get("epoch", 0) or 0), row


def best_threshold_from_rows(rows: list[dict[str, Any]]) -> float:
    _, row = best_row(rows, "best_f1")
    return to_float(row.get("valid_best_threshold"), 0.0)


def metric_best(rows: list[dict[str, Any]], metric: str, mode: str = "max") -> float:
    _, row = best_row(rows, metric, mode=mode)
    return to_float(row.get(f"valid_{metric}"), 0.0)


def extract_by_type(row: dict[str, Any], run_summary: dict[str, Any]) -> list[dict[str, Any]]:
    prefix = "valid_by_type_"
    metrics_by_type: dict[str, dict[str, Any]] = {}
    for key, value in row.items():
        if not key.startswith(prefix):
            continue
        suffixes = ["_roc_auc", "_f1", "_pr_auc", "_accuracy", "_count_positive", "_count_negative"]
        for suffix in suffixes:
            if key.endswith(suffix):
                neg_type = key[len(prefix):-len(suffix)]
                metric_name = suffix.strip("_")
                metrics_by_type.setdefault(neg_type, {})[metric_name] = value
                break
    out = []
    for neg_type, values in sorted(metrics_by_type.items()):
        out.append(
            {
                "run_name": run_summary["run_name"],
                "family": run_summary["family"],
                "image_encoder_name": run_summary["image_encoder_name"],
                "best_epoch": run_summary["best_epoch"],
                "negative_type": neg_type,
                "roc_auc": to_float(values.get("roc_auc")),
                "f1": to_float(values.get("f1")),
                "pr_auc": to_float(values.get("pr_auc")),
                "accuracy": to_float(values.get("accuracy")),
                "count_positive": to_float(values.get("count_positive")),
                "count_negative": to_float(values.get("count_negative")),
                "run_dir": run_summary["run_dir"],
            }
        )
    return out


def collect_run(run_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    config = read_yaml(run_dir / "config_snapshot.yaml")
    summary = read_json(run_dir / "summary.json")
    status = read_json(run_dir / "status.json")
    metrics_rows = read_jsonl(run_dir / "metrics.jsonl")
    best_epoch, roc_row = best_row(metrics_rows, "roc_auc")
    include_negative_types = (config.get("data") or {}).get("include_negative_types") or []
    run_name = str(summary.get("run_name") or config.get("run_name") or status.get("run_name") or run_dir.name)
    family = infer_family(config, run_dir)
    epochs_completed = int(summary.get("epochs_completed") or len(metrics_rows) or 0)
    row = {
        "run_name": run_name,
        "status": str(summary.get("status") or status.get("status") or ("RUNNING" if metrics_rows else "UNKNOWN")),
        "family": family,
        "image_encoder_name": str(config.get("image_encoder_name") or config.get("model_type") or ""),
        "text_encoder_name": str(config.get("text_encoder_name") or ""),
        "tokenizer_name": str(config.get("tokenizer_name") or ""),
        "pairs_csv": str(config.get("pairs_csv") or ""),
        "include_negative_types": include_negative_types,
        "epochs_completed": epochs_completed,
        "best_epoch": best_epoch or summary.get("early_stopping_best_epoch") or epochs_completed,
        "best_valid_roc_auc": to_float(summary.get("best_valid_roc_auc"), metric_best(metrics_rows, "roc_auc")),
        "best_valid_f1": to_float(summary.get("best_valid_f1"), metric_best(metrics_rows, "best_f1")),
        "best_valid_threshold": to_float(summary.get("best_threshold"), best_threshold_from_rows(metrics_rows)),
        "best_valid_pr_auc": metric_best(metrics_rows, "pr_auc"),
        "best_valid_loss": metric_best(metrics_rows, "loss", mode="min"),
        "peak_vram_mb": to_float(summary.get("peak_vram_mb")),
        "runtime_min": to_float(summary.get("runtime_min"), to_float(status.get("runtime_min"))),
        "collapse_detected": bool(summary.get("collapse_detected", False)),
        "batch_size": config.get("batch_size"),
        "learning_rate": config.get("learning_rate"),
        "weight_decay": config.get("weight_decay"),
        "dropout": config.get("dropout"),
        "scheduler_type": config.get("scheduler_type"),
        "warmup_ratio": config.get("warmup_ratio"),
        "max_train_samples": config.get("max_train_samples"),
        "max_valid_samples": config.get("max_valid_samples"),
        "rank_roc_auc": None,
        "rank_f1": None,
        "run_dir": str(run_dir),
    }
    by_type = extract_by_type(roc_row, row)
    return row, by_type


def add_ranks(rows: list[dict[str, Any]]) -> None:
    roc_sorted = sorted(rows, key=lambda row: to_float(row.get("best_valid_roc_auc"), -math.inf), reverse=True)
    f1_sorted = sorted(rows, key=lambda row: to_float(row.get("best_valid_f1"), -math.inf), reverse=True)
    for rank, row in enumerate(roc_sorted, start=1):
        row["rank_roc_auc"] = rank
    for rank, row in enumerate(f1_sorted, start=1):
        row["rank_f1"] = rank


def sort_rows(rows: list[dict[str, Any]], sort_by: str) -> list[dict[str, Any]]:
    if sort_by == "f1":
        return sorted(rows, key=lambda row: to_float(row.get("best_valid_f1")), reverse=True)
    if sort_by == "pr_auc":
        return sorted(rows, key=lambda row: to_float(row.get("best_valid_pr_auc")), reverse=True)
    if sort_by == "loss":
        return sorted(rows, key=lambda row: to_float(row.get("best_valid_loss"), math.inf))
    if sort_by == "runtime":
        return sorted(rows, key=lambda row: to_float(row.get("runtime_min"), math.inf))
    if sort_by == "vram":
        return sorted(rows, key=lambda row: to_float(row.get("peak_vram_mb"), math.inf))
    return sorted(rows, key=lambda row: to_float(row.get("best_valid_roc_auc")), reverse=True)


def best_by_family(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for family in sorted({row.get("family") for row in rows}):
        candidates = [row for row in rows if row.get("family") == family and row.get("status") == "OK" and not row.get("collapse_detected")]
        if not candidates:
            continue
        out.append(max(candidates, key=lambda row: (to_float(row.get("best_valid_roc_auc")), to_float(row.get("best_valid_f1")))))
    return sorted(out, key=lambda row: to_float(row.get("best_valid_roc_auc")), reverse=True)


def print_preview(rows: list[dict[str, Any]], top_k: int) -> None:
    preview = rows[:top_k] if top_k > 0 else rows[:10]
    print("run_name                       status     family    encoder            auc     f1      epoch")
    for row in preview:
        print(
            f"{str(row['run_name'])[:30]:30} {str(row['status'])[:10]:10} {str(row['family'])[:9]:9} "
            f"{str(row['image_encoder_name'])[:18]:18} {to_float(row['best_valid_roc_auc']):.4f} "
            f"{to_float(row['best_valid_f1']):.4f} {row['best_epoch']}"
        )


def main() -> None:
    args = parse_args()
    run_dirs = discover_run_dirs(args.results_root, include_running=args.include_running)
    if args.dry_run:
        completed = sum(1 for run_dir in run_dirs if (run_dir / "summary.json").exists())
        print(f"Discovered run_dirs={len(run_dirs)} completed_with_summary={completed} results_root={args.results_root}")
        return
    rows = []
    by_type_rows = []
    for run_dir in run_dirs:
        row, by_type = collect_run(run_dir)
        rows.append(row)
        by_type_rows.extend(by_type)
    add_ranks(rows)
    rows = sort_rows(rows, args.sort_by)
    best_rows = best_by_family(rows)
    write_csv(args.out_dir / "runs_summary.csv", rows, SUMMARY_FIELDS)
    write_json(args.out_dir / "runs_summary.json", rows)
    write_csv(args.out_dir / "by_type_metrics.csv", by_type_rows, BY_TYPE_FIELDS)
    write_csv(args.out_dir / "best_by_family.csv", best_rows, BEST_BY_FAMILY_FIELDS)
    print(f"Wrote {len(rows)} runs to {args.out_dir}")
    print_preview(rows, args.top_k)


if __name__ == "__main__":
    main()

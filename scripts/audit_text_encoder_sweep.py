from __future__ import annotations

from pathlib import Path
import argparse
import csv
import json
import math
import sys

import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_ROOT = ROOT / "experiments" / "text_sweeps"
DEFAULT_OUT_DIR = ROOT / "results" / "text_sweeps"
SUMMARY_FIELDS = [
    "run_name", "family", "image_encoder", "text_encoder", "freeze_text_encoder",
    "status", "best_epoch", "best_valid_roc_auc", "best_valid_f1", "best_threshold",
    "runtime_min", "peak_vram_mb", "collapse_detected", "run_dir",
]
BY_TYPE_FIELDS = ["run_name", "image_encoder", "text_encoder", "freeze_text_encoder", "negative_type", "roc_auc"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit controlled text encoder sweep results.")
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--include-running", action="store_true")
    parser.add_argument("--sort-by", choices=["roc_auc", "f1", "runtime", "vram"], default="roc_auc")
    return parser.parse_args()


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open() as f:
        return json.load(f)


def read_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open() as f:
        return yaml.safe_load(f) or {}


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


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


def to_float(value, default=0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def infer_family(config: dict) -> str:
    if config.get("run_family"):
        return str(config["run_family"])
    image = str(config.get("image_encoder_name", "")).lower()
    if image.startswith("convnext"):
        return "convnext"
    if image.startswith("deit"):
        return "deit"
    if image.startswith("vit"):
        return "vit"
    return "unknown"


def discover_run_dirs(root: Path, include_running: bool) -> list[Path]:
    if not root.exists():
        return []
    run_dirs = {p.parent for p in root.rglob("*") if p.is_file() and p.name in {"summary.json", "metrics.jsonl", "config_snapshot.yaml"}}
    if not include_running:
        run_dirs = {d for d in run_dirs if (d / "summary.json").exists()}
    return sorted(run_dirs)


def best_epoch_row(rows: list[dict]) -> tuple[int | None, dict]:
    if not rows:
        return None, {}
    candidates = [row for row in rows if "valid_roc_auc" in row]
    if not candidates:
        return None, rows[-1]
    row = max(candidates, key=lambda item: to_float(item.get("valid_roc_auc"), -math.inf))
    return int(row.get("epoch", 0) or 0), row


def by_type_auc(row: dict, summary: dict) -> list[dict]:
    out = []
    prefix = "valid_by_type_"
    suffix = "_roc_auc"
    for key, value in sorted(row.items()):
        if key.startswith(prefix) and key.endswith(suffix):
            out.append({
                "run_name": summary["run_name"],
                "image_encoder": summary["image_encoder"],
                "text_encoder": summary["text_encoder"],
                "freeze_text_encoder": summary["freeze_text_encoder"],
                "negative_type": key[len(prefix):-len(suffix)],
                "roc_auc": to_float(value),
            })
    return out


def collect_run(run_dir: Path) -> tuple[dict, list[dict]]:
    config = read_yaml(run_dir / "config_snapshot.yaml")
    summary = read_json(run_dir / "summary.json")
    status = read_json(run_dir / "status.json")
    metrics = read_jsonl(run_dir / "metrics.jsonl")
    best_epoch, row = best_epoch_row(metrics)
    run_name = str(summary.get("run_name") or config.get("run_name") or run_dir.name)
    out = {
        "run_name": run_name,
        "family": infer_family(config),
        "image_encoder": str(config.get("image_encoder_name") or ""),
        "text_encoder": str(config.get("text_encoder_name") or ""),
        "freeze_text_encoder": bool(config.get("freeze_text_encoder", False)),
        "status": str(summary.get("status") or status.get("status") or "RUNNING"),
        "best_epoch": best_epoch or summary.get("early_stopping_best_epoch") or len(metrics),
        "best_valid_roc_auc": to_float(summary.get("best_valid_roc_auc"), to_float(row.get("valid_roc_auc"))),
        "best_valid_f1": to_float(summary.get("best_valid_f1"), to_float(row.get("valid_best_f1"))),
        "best_threshold": to_float(summary.get("best_threshold"), to_float(row.get("valid_best_threshold"))),
        "runtime_min": to_float(summary.get("runtime_min"), to_float(status.get("runtime_min"))),
        "peak_vram_mb": to_float(summary.get("peak_vram_mb")),
        "collapse_detected": bool(summary.get("collapse_detected", False)),
        "run_dir": str(run_dir),
    }
    return out, by_type_auc(row, out)


def sort_rows(rows: list[dict], sort_by: str) -> list[dict]:
    if sort_by == "f1":
        return sorted(rows, key=lambda r: to_float(r["best_valid_f1"]), reverse=True)
    if sort_by == "runtime":
        return sorted(rows, key=lambda r: to_float(r["runtime_min"], math.inf))
    if sort_by == "vram":
        return sorted(rows, key=lambda r: to_float(r["peak_vram_mb"], math.inf))
    return sorted(rows, key=lambda r: to_float(r["best_valid_roc_auc"]), reverse=True)


def by_type_text(rows: list[dict], run_name: str) -> str:
    subset = [r for r in rows if r["run_name"] == run_name]
    if not subset:
        return "-"
    return ", ".join(f"{r['negative_type']}:{to_float(r['roc_auc']):.3f}" for r in subset)


def main() -> None:
    args = parse_args()
    run_dirs = discover_run_dirs(args.results_root, args.include_running)
    if not run_dirs:
        print(f"No text sweep results found under {args.results_root}")
        return
    summaries = []
    by_types = []
    for run_dir in run_dirs:
        summary, by_type = collect_run(run_dir)
        summaries.append(summary)
        by_types.extend(by_type)
    summaries = sort_rows(summaries, args.sort_by)
    write_csv(args.out_dir / "runs_summary.csv", summaries, SUMMARY_FIELDS)
    write_json(args.out_dir / "runs_summary.json", summaries)
    write_csv(args.out_dir / "by_type_auc.csv", by_types, BY_TYPE_FIELDS)
    print(f"Wrote text sweep audit to {args.out_dir}")
    print("run_name                              image          text           freeze epoch auc     f1      thr   runtime vram collapse")
    for row in summaries:
        print(f"{row['run_name'][:36]:36} {row['image_encoder'][:14]:14} {row['text_encoder'][:14]:14} {str(row['freeze_text_encoder']):6} {str(row['best_epoch']):>5} {to_float(row['best_valid_roc_auc']):.4f} {to_float(row['best_valid_f1']):.4f} {to_float(row['best_threshold']):.2f} {to_float(row['runtime_min']):7.1f} {to_float(row['peak_vram_mb']):5.0f} {str(row['collapse_detected']).lower()}")
        print(f"  by-type AUC: {by_type_text(by_types, row['run_name'])}")


if __name__ == "__main__":
    main()

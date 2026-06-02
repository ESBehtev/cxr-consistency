from __future__ import annotations

from pathlib import Path
import argparse
import csv
import json
import math

try:
    import yaml
except ModuleNotFoundError:  # pragma: no cover - used only in minimal shells
    yaml = None

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_ROOT = ROOT / "experiments" / "final_full"
DEFAULT_OUT_DIR = ROOT / "results" / "final_full"
NEG_TYPES = [
    "pathology_matched_report",
    "laterality_conflict",
    "distorted_negation",
    "partial_mismatch",
    "pathology_semantic_swap",
    "temporal_mismatch",
]

FIELDS = [
    "run_name", "family", "image_encoder", "text_encoder", "status", "epochs_completed",
    "best_epoch", "best_valid_roc_auc", "best_valid_f1", "best_threshold",
    "runtime_min", "peak_vram_mb", "collapse_detected", "hardest_negative_type", "hardest_negative_auc",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Read-only audit for final full runs.")
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--include-running", action="store_true")
    return parser.parse_args()


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open() as f:
        return json.load(f)


def parse_scalar(value: str):
    if value == "null":
        return None
    if value == "true":
        return True
    if value == "false":
        return False
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value

def read_simple_yaml(path: Path) -> dict:
    root: dict = {}
    stack: list[tuple[int, dict]] = [(-1, root)]
    pending_list: tuple[int, dict, str] | None = None
    with path.open() as f:
        for raw in f:
            if not raw.strip() or raw.lstrip().startswith("#"):
                continue
            indent = len(raw) - len(raw.lstrip(" "))
            line = raw.strip()
            if line.startswith("- "):
                if pending_list is None:
                    raise ValueError(f"Unsupported YAML list in {path}: {line}")
                list_indent, parent, key = pending_list
                if indent < list_indent:
                    pending_list = None
                    raise ValueError(f"Unsupported YAML list indentation in {path}: {line}")
                if not isinstance(parent.get(key), list):
                    parent[key] = []
                parent[key].append(parse_scalar(line[2:].strip()))
                continue
            pending_list = None
            key, sep, value = line.partition(":")
            if not sep:
                raise ValueError(f"Unsupported YAML line in {path}: {line}")
            while stack and indent <= stack[-1][0]:
                stack.pop()
            parent = stack[-1][1]
            key = key.strip()
            value = value.strip()
            if value == "":
                child: dict = {}
                parent[key] = child
                stack.append((indent, child))
                pending_list = (indent, parent, key)
            else:
                parent[key] = parse_scalar(value)
    return root

def read_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    if yaml is not None:
        with path.open() as f:
            return yaml.safe_load(f) or {}
    return read_simple_yaml(path)


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


def as_float(value, default=0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def infer_family(cfg: dict) -> str:
    if cfg.get("run_family"):
        return str(cfg["run_family"])
    image = str(cfg.get("image_encoder_name") or "").lower()
    if image.startswith("convnext"):
        return "convnext"
    if image.startswith("deit"):
        return "deit"
    if image.startswith("vit"):
        return "vit"
    return "unknown"


def text_name(cfg: dict) -> str:
    if cfg.get("text_encoder_name") == "cxrbert":
        return "cxrbert_frozen" if cfg.get("freeze_text_encoder") else "cxrbert_unfrozen"
    return str(cfg.get("text_encoder_name") or "")


def best_row(metrics: list[dict]) -> dict:
    valid = [row for row in metrics if "valid_roc_auc" in row]
    if valid:
        return max(valid, key=lambda row: as_float(row.get("valid_roc_auc"), -math.inf))
    return metrics[-1] if metrics else {}


def hardest(row: dict) -> tuple[str, float | str]:
    vals = []
    for neg in NEG_TYPES:
        key = f"valid_by_type_{neg}_roc_auc"
        if key in row:
            vals.append((neg, as_float(row[key], math.inf)))
    if not vals:
        return "", ""
    return min(vals, key=lambda item: item[1])


def discover(root: Path, include_running: bool) -> list[Path]:
    if not root.exists():
        return []
    runs = [p for p in root.iterdir() if p.is_dir()]
    if include_running:
        return sorted(runs)
    return sorted(p for p in runs if (p / "summary.json").exists())


def collect(run_dir: Path) -> dict:
    cfg = read_yaml(run_dir / "config_snapshot.yaml")
    status = read_json(run_dir / "status.json")
    summary = read_json(run_dir / "summary.json")
    metrics = read_jsonl(run_dir / "metrics.jsonl")
    best = best_row(metrics)
    hard_name, hard_auc = hardest(best)
    run_name = str(summary.get("run_name") or cfg.get("run_name") or status.get("run_name") or run_dir.name)
    return {
        "run_name": run_name,
        "family": infer_family(cfg),
        "image_encoder": str(cfg.get("image_encoder_name") or cfg.get("model_type") or ""),
        "text_encoder": text_name(cfg),
        "status": str(summary.get("status") or status.get("status") or "RUNNING"),
        "epochs_completed": int(summary.get("epochs_completed") or len(metrics) or 0),
        "best_epoch": summary.get("early_stopping_best_epoch") or best.get("epoch") or "",
        "best_valid_roc_auc": as_float(summary.get("best_valid_roc_auc"), as_float(best.get("valid_roc_auc"))),
        "best_valid_f1": as_float(summary.get("best_valid_f1"), as_float(best.get("valid_best_f1"))),
        "best_threshold": as_float(summary.get("best_threshold"), as_float(best.get("valid_best_threshold"))),
        "runtime_min": as_float(summary.get("runtime_min"), as_float(status.get("runtime_min"))),
        "peak_vram_mb": as_float(summary.get("peak_vram_mb")),
        "collapse_detected": bool(summary.get("collapse_detected", False)),
        "hardest_negative_type": hard_name,
        "hardest_negative_auc": hard_auc,
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_json(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(rows, f, indent=2, sort_keys=True)
        f.write("\n")


def main() -> None:
    args = parse_args()
    rows = [collect(run_dir) for run_dir in discover(args.results_root, args.include_running)]
    rows.sort(key=lambda row: (as_float(row["best_valid_roc_auc"]), as_float(row["best_valid_f1"])), reverse=True)
    if not rows:
        print(f"No final full results found under {args.results_root}")
        return
    write_csv(args.out_dir / "runs_summary.csv", rows)
    write_json(args.out_dir / "runs_summary.json", rows)
    print(f"Wrote final full audit to {args.out_dir}")
    print("run_name                     status     epochs epoch auc     f1      thr   runtime vram collapse hardest")
    for row in rows:
        print(
            f"{row['run_name'][:28]:28} {row['status'][:10]:10} {row['epochs_completed']:>6} {str(row['best_epoch']):>5} "
            f"{as_float(row['best_valid_roc_auc']):.4f} {as_float(row['best_valid_f1']):.4f} {as_float(row['best_threshold']):.2f} "
            f"{as_float(row['runtime_min']):7.1f} {as_float(row['peak_vram_mb']):5.0f} {str(row['collapse_detected']).lower():8} "
            f"{row['hardest_negative_type']}:{row['hardest_negative_auc']}"
        )

if __name__ == "__main__":
    main()

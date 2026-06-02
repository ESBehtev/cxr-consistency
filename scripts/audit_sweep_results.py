from __future__ import annotations

from pathlib import Path
import argparse
import json
import math

import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_ROOT = ROOT / "experiments" / "sweeps"
DEFAULT_CONFIG_ROOT = ROOT / "configs" / "sweeps"
FAMILIES = ["convnext", "vit", "deit"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit local sweep results.")
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--configs-root", type=Path, default=DEFAULT_CONFIG_ROOT)
    parser.add_argument("--sort-by", choices=["roc_auc", "f1", "runtime", "vram"], default="roc_auc")
    return parser.parse_args()


def read_json(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def read_yaml(path: Path) -> dict:
    with path.open() as f:
        return yaml.safe_load(f)


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


def infer_family(run_dir: Path, config: dict) -> str:
    family = config.get("run_family")
    if family:
        return str(family)
    model = str(config.get("model_type") or config.get("image_encoder_name") or "").lower()
    if model.startswith("convnext"):
        return "convnext"
    if model.startswith("deit"):
        return "deit"
    if model.startswith("vit"):
        return "vit"
    return "unknown"


def metric_at_best_epoch(metrics_rows: list[dict]) -> tuple[int | None, dict]:
    if not metrics_rows:
        return None, {}
    best = max(metrics_rows, key=lambda row: float(row.get("valid_roc_auc", -math.inf)))
    return int(best.get("epoch", 0)), best


def by_type_auc(row: dict) -> dict[str, float]:
    out = {}
    prefix = "valid_by_type_"
    suffix = "_roc_auc"
    for key, value in row.items():
        if key.startswith(prefix) and key.endswith(suffix):
            neg_type = key[len(prefix):-len(suffix)]
            out[neg_type] = float(value)
    return out


def format_by_type(aucs: dict[str, float]) -> str:
    if not aucs:
        return "-"
    return ", ".join(f"{k}:{v:.3f}" for k, v in sorted(aucs.items()))


def hardest_type(aucs: dict[str, float]) -> tuple[str, float] | tuple[None, None]:
    if not aucs:
        return None, None
    return min(aucs.items(), key=lambda item: item[1])


def load_runs(results_root: Path) -> list[dict]:
    runs = []
    if not results_root.exists():
        return runs
    for run_dir in sorted(p for p in results_root.iterdir() if p.is_dir()):
        summary_path = run_dir / "summary.json"
        metrics_path = run_dir / "metrics.jsonl"
        config_path = run_dir / "config_snapshot.yaml"
        status_path = run_dir / "status.json"
        if not summary_path.exists() and not status_path.exists():
            continue
        summary = read_json(summary_path) if summary_path.exists() else {}
        config = read_yaml(config_path) if config_path.exists() else {}
        status = read_json(status_path) if status_path.exists() else {}
        metrics_rows = read_jsonl(metrics_path)
        best_epoch, best_row = metric_at_best_epoch(metrics_rows)
        aucs = by_type_auc(best_row)
        hard_name, hard_auc = hardest_type(aucs)
        run_name = str(summary.get("run_name") or config.get("run_name") or run_dir.name)
        family = infer_family(run_dir, config)
        runs.append(
            {
                "run_name": run_name,
                "family": family,
                "encoder": str(config.get("image_encoder_name") or config.get("model_type") or "unknown"),
                "status": str(summary.get("status") or status.get("status") or "UNKNOWN"),
                "best_epoch": best_epoch or summary.get("early_stopping_best_epoch"),
                "best_valid_roc_auc": float(summary.get("best_valid_roc_auc", best_row.get("valid_roc_auc", 0.0) if best_row else 0.0)),
                "best_valid_f1": float(summary.get("best_valid_f1", best_row.get("valid_best_f1", 0.0) if best_row else 0.0)),
                "threshold": float(summary.get("best_threshold", best_row.get("valid_best_threshold", 0.0) if best_row else 0.0)),
                "runtime_min": float(summary.get("runtime_min", status.get("runtime_min", 0.0) or 0.0)),
                "peak_vram_mb": float(summary.get("peak_vram_mb", 0.0) or 0.0),
                "collapse": bool(summary.get("collapse_detected", False)),
                "hardest_negative_type": hard_name,
                "hardest_negative_auc": hard_auc,
                "by_type_auc": aucs,
                "run_dir": run_dir,
                "config": config,
            }
        )
    return runs


def sort_runs(runs: list[dict], sort_by: str) -> list[dict]:
    if sort_by == "f1":
        return sorted(runs, key=lambda r: r["best_valid_f1"], reverse=True)
    if sort_by == "runtime":
        return sorted(runs, key=lambda r: r["runtime_min"])
    if sort_by == "vram":
        return sorted(runs, key=lambda r: r["peak_vram_mb"])
    return sorted(runs, key=lambda r: r["best_valid_roc_auc"], reverse=True)


def print_table(runs: list[dict]) -> None:
    print("run_name                       family    encoder            status     epoch auc     f1      thr   runtime vram  collapse hardest")
    for r in runs:
        hard = "-" if r["hardest_negative_type"] is None else f"{r['hardest_negative_type']}:{r['hardest_negative_auc']:.3f}"
        print(
            f"{r['run_name'][:30]:30} {r['family'][:9]:9} {r['encoder'][:18]:18} "
            f"{r['status'][:10]:10} {str(r['best_epoch'] or '-'):>5} "
            f"{r['best_valid_roc_auc']:.4f} {r['best_valid_f1']:.4f} {r['threshold']:.2f} "
            f"{r['runtime_min']:7.1f} {r['peak_vram_mb']:5.0f} {str(r['collapse']).lower():8} {hard}"
        )


def best_run(runs: list[dict], family: str | None = None) -> dict | None:
    candidates = [r for r in runs if r["status"] == "OK" and not r["collapse"]]
    if family is not None:
        candidates = [r for r in candidates if r["family"] == family]
    if not candidates:
        return None
    return max(candidates, key=lambda r: (r["best_valid_roc_auc"], r["best_valid_f1"]))


def safest_run(runs: list[dict]) -> dict | None:
    candidates = [r for r in runs if r["status"] == "OK" and not r["collapse"]]
    if not candidates:
        return None
    learned = [r for r in candidates if r["best_valid_roc_auc"] >= 0.65]
    pool = learned or candidates
    return min(pool, key=lambda r: (r["peak_vram_mb"], r["runtime_min"], -r["best_valid_roc_auc"]))


def print_recommendations(runs: list[dict]) -> None:
    print("\nBest by family")
    for family in FAMILIES:
        r = best_run(runs, family)
        if r is None:
            print(f"{family}: -")
        else:
            print(f"{family}: {r['run_name']} auc={r['best_valid_roc_auc']:.4f} f1={r['best_valid_f1']:.4f} epoch={r['best_epoch']}")
    overall = best_run(runs)
    safest = safest_run(runs)
    print("\nBest overall")
    print("-" if overall is None else f"{overall['run_name']} auc={overall['best_valid_roc_auc']:.4f} f1={overall['best_valid_f1']:.4f}")
    print("\nSafest config")
    print("-" if safest is None else f"{safest['run_name']} peak_vram_mb={safest['peak_vram_mb']:.0f} runtime_min={safest['runtime_min']:.1f}")
    print("\nRecommendation for coursework final runs")
    if overall is None:
        print("Run the sweep first, then rerun the best ConvNext and best transformer config with the canonical full baseline budget.")
    else:
        print(
            f"Use {overall['run_name']} as the primary candidate, compare against the best family baselines above, "
            "and rerun finalists with the canonical hard-pair validation budget before reporting final metrics."
        )


def print_pending_configs(configs_root: Path) -> None:
    configs = sorted(configs_root.glob("**/*.yaml"))
    if not configs:
        print("No sweep configs found.")
        return
    print(f"No completed sweep results found. Pending configs: {len(configs)}")
    for family in FAMILIES:
        family_configs = [p for p in configs if p.parent.name == family]
        print(f"{family}: {len(family_configs)}")


def main() -> None:
    args = parse_args()
    runs = load_runs(args.results_root)
    if not runs:
        print_pending_configs(args.configs_root)
        return
    runs = sort_runs(runs, args.sort_by)
    print_table(runs)
    print("\nBy-negative-type AUC at best ROC-AUC epoch")
    for r in runs:
        print(f"{r['run_name']}: {format_by_type(r['by_type_auc'])}")
    print_recommendations(runs)


if __name__ == "__main__":
    main()

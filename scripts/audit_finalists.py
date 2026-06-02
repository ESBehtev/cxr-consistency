from __future__ import annotations

from pathlib import Path
import argparse
import json
import math
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[0]
sys.path.insert(0, str(SCRIPT_DIR))

from collect_experiment_results import (  # noqa: E402
    SUMMARY_FIELDS,
    BY_TYPE_FIELDS,
    BEST_BY_FAMILY_FIELDS,
    add_ranks,
    best_by_family,
    collect_run,
    discover_run_dirs,
    sort_rows,
    to_float,
    write_csv,
    write_json,
)


DEFAULT_RESULTS_ROOT = ROOT / "experiments" / "finalists"
DEFAULT_OUT_DIR = ROOT / "results" / "finalists"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit final-stage local runs.")
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--sort-by", choices=["roc_auc", "f1", "pr_auc", "loss", "runtime", "vram"], default="roc_auc")
    parser.add_argument("--include-running", action="store_true")
    return parser.parse_args()


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


def overfit_summary(run_dir: str) -> str:
    rows = read_jsonl(Path(run_dir) / "metrics.jsonl")
    if len(rows) < 2:
        return "insufficient_epochs"
    best_valid_auc = max(to_float(row.get("valid_roc_auc"), -math.inf) for row in rows)
    last = rows[-1]
    last_valid_auc = to_float(last.get("valid_roc_auc"), 0.0)
    last_train_auc = to_float(last.get("train_roc_auc"), 0.0)
    last_valid_loss = to_float(last.get("valid_loss"), 0.0)
    best_valid_loss = min(to_float(row.get("valid_loss"), math.inf) for row in rows)
    flags = []
    if last_train_auc - last_valid_auc > 0.08:
        flags.append(f"train_valid_auc_gap={last_train_auc - last_valid_auc:.3f}")
    if best_valid_auc - last_valid_auc > 0.02:
        flags.append(f"valid_auc_drop={best_valid_auc - last_valid_auc:.3f}")
    if last_valid_loss - best_valid_loss > 0.05:
        flags.append(f"valid_loss_rise={last_valid_loss - best_valid_loss:.3f}")
    return "ok" if not flags else "; ".join(flags)


def by_type_auc_text(rows: list[dict], run_name: str) -> str:
    subset = [row for row in rows if row["run_name"] == run_name]
    if not subset:
        return "-"
    return ", ".join(f"{row['negative_type']}:{to_float(row.get('roc_auc')):.3f}" for row in sorted(subset, key=lambda item: item["negative_type"]))


def main() -> None:
    args = parse_args()
    run_dirs = discover_run_dirs(args.results_root, include_running=args.include_running)
    if not run_dirs:
        print(f"No finalist results found under {args.results_root}")
        return
    rows = []
    by_type_rows = []
    for run_dir in run_dirs:
        row, by_type = collect_run(run_dir)
        row["overfit_summary"] = overfit_summary(row["run_dir"])
        rows.append(row)
        by_type_rows.extend(by_type)
    add_ranks(rows)
    rows = sort_rows(rows, args.sort_by)
    best_rows = best_by_family(rows)
    fields = SUMMARY_FIELDS + ["overfit_summary"]
    write_csv(args.out_dir / "runs_summary.csv", rows, fields)
    write_json(args.out_dir / "runs_summary.json", rows)
    write_csv(args.out_dir / "by_type_metrics.csv", by_type_rows, BY_TYPE_FIELDS)
    write_csv(args.out_dir / "best_by_family.csv", best_rows, BEST_BY_FAMILY_FIELDS + ["overfit_summary"])
    print(f"Wrote finalist audit to {args.out_dir}")
    print("run_name                       status     epoch auc     f1      runtime vram  collapse overfit")
    for row in rows:
        print(
            f"{str(row['run_name'])[:30]:30} {str(row['status'])[:10]:10} {str(row['best_epoch']):>5} "
            f"{to_float(row['best_valid_roc_auc']):.4f} {to_float(row['best_valid_f1']):.4f} "
            f"{to_float(row['runtime_min']):7.1f} {to_float(row['peak_vram_mb']):5.0f} "
            f"{str(row['collapse_detected']).lower():8} {row['overfit_summary']}"
        )
        print(f"  by-type AUC: {by_type_auc_text(by_type_rows, row['run_name'])}")


if __name__ == "__main__":
    main()

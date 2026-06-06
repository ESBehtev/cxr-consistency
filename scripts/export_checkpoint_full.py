from __future__ import annotations

from pathlib import Path
import argparse
import csv
import json
import shutil

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_ROOT = ROOT / "experiments" / "checkpoint_full"
DEFAULT_AUDIT_DIR = ROOT / "results" / "checkpoint_full"
DEFAULT_EXPORT_DIR = ROOT / "results" / "coursework_export" / "checkpoint_full"
LIGHT_FILES = ["config_snapshot.yaml", "metrics.jsonl", "summary.json", "status.json", "train.log"]
CHECKPOINT_FILES = ["best_model.pt", "last_model.pt"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export checkpoint_full run artifacts.")
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--audit-dir", type=Path, default=DEFAULT_AUDIT_DIR)
    parser.add_argument("--export-dir", type=Path, default=DEFAULT_EXPORT_DIR)
    parser.add_argument("--skip-checkpoints", action="store_true")
    parser.add_argument("--include-running", action="store_true")
    return parser.parse_args()


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open() as f:
        return json.load(f)


def run_name(run_dir: Path) -> str:
    summary = read_json(run_dir / "summary.json")
    status = read_json(run_dir / "status.json")
    return str(summary.get("run_name") or status.get("run_name") or run_dir.name)


def discover(root: Path, include_running: bool) -> list[Path]:
    if not root.exists():
        return []
    runs = [p for p in root.iterdir() if p.is_dir()]
    if include_running:
        return sorted(runs)
    return sorted(p for p in runs if (p / "summary.json").exists())


def copy_file(src: Path, dst: Path) -> bool:
    if not src.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def copy_config_source(run_dir: Path, dst_dir: Path) -> bool:
    status = read_json(run_dir / "status.json")
    config_path = status.get("config_path")
    if not config_path:
        return False
    src = Path(config_path)
    if not src.is_absolute():
        src = ROOT / src
    return copy_file(src, dst_dir / src.name)


def write_manifest(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(rows, f, indent=2, sort_keys=True)
        f.write("\n")


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fields = sorted({key for row in rows for key in row})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    runs = discover(args.results_root, args.include_running)
    args.export_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for run_dir in runs:
        name = run_name(run_dir)
        dst = args.export_dir / name
        copied = []
        copy_config_source(run_dir, dst)
        for filename in LIGHT_FILES:
            if copy_file(run_dir / filename, dst / filename):
                copied.append(filename)
        if not args.skip_checkpoints:
            for filename in CHECKPOINT_FILES:
                if copy_file(run_dir / filename, dst / filename):
                    copied.append(filename)
        manifest.append({"run_name": name, "source": str(run_dir), "destination": str(dst), "copied": copied, "checkpoints_included": not args.skip_checkpoints})
    for filename in ["runs_summary.csv", "runs_summary.json"]:
        copy_file(args.audit_dir / filename, args.export_dir / filename)
    write_manifest(args.export_dir / "export_manifest.json", manifest)
    write_csv(args.export_dir / "export_manifest.csv", manifest)
    print(f"Exported checkpoint_full artifacts: runs={len(runs)} to {args.export_dir}")
    if args.skip_checkpoints:
        print("Skipped checkpoint .pt files")


if __name__ == "__main__":
    main()

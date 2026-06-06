from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import argparse
import json
import subprocess
import time

try:
    import yaml
except ModuleNotFoundError:
    yaml = None

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_ROOT = ROOT / "configs" / "checkpoint_full"
DEFAULT_OUTPUT_ROOT = ROOT / "experiments" / "checkpoint_full"
STATUS_JSON = "checkpoint_full_status.json"
STATUS_JSONL = "checkpoint_full_status.jsonl"

@dataclass
class Job:
    config_path: Path
    run_name: str
    batch_size: int
    expected_vram_mb: int
    run_dir: Path
    status: str = "PENDING"
    started_at: float | None = None
    finished_at: float | None = None
    return_code: int | None = None
    summary: dict = field(default_factory=dict)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Single-config launcher for checkpoint_full training.")
    parser.add_argument("--config", type=Path, help="Config to run. Required unless --dry-run is used.")
    parser.add_argument("--configs-root", type=Path, default=DEFAULT_CONFIG_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--python", type=str, default=str(ROOT / ".venv" / "bin" / "python"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--rerun-completed", action="store_true")
    return parser.parse_args()


def now() -> float:
    return time.time()


def read_yaml(path: Path) -> dict:
    if yaml is None:
        raise RuntimeError("PyYAML is required for this launcher")
    with path.open() as f:
        return yaml.safe_load(f) or {}


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open() as f:
        return json.load(f)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


def append_jsonl(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        json.dump(payload, f, sort_keys=True)
        f.write("\n")


def build_job(config_path: Path) -> Job:
    config_path = config_path.resolve()
    cfg = read_yaml(config_path)
    run_name = str(cfg.get("run_name") or config_path.stem)
    run_dir = Path(cfg.get("experiment_dir") or DEFAULT_OUTPUT_ROOT / run_name)
    if not run_dir.is_absolute():
        run_dir = ROOT / run_dir
    resources = cfg.get("resources", {}) or {}
    return Job(
        config_path=config_path,
        run_name=run_name,
        batch_size=int(cfg.get("batch_size", 0)),
        expected_vram_mb=int(resources.get("expected_vram_mb", 0)),
        run_dir=run_dir,
    )


def payload(job: Job) -> dict:
    runtime = None
    if job.started_at is not None:
        runtime = ((job.finished_at or now()) - job.started_at) / 60.0
    return {
        "run_name": job.run_name,
        "config_path": str(job.config_path),
        "run_dir": str(job.run_dir),
        "batch_size": job.batch_size,
        "expected_vram_mb": job.expected_vram_mb,
        "status": job.status,
        "started_at": job.started_at,
        "finished_at": job.finished_at,
        "runtime_min": runtime,
        "return_code": job.return_code,
        "best_epoch": job.summary.get("best_epoch") or job.summary.get("early_stopping_best_epoch"),
        "best_valid_roc_auc": job.summary.get("best_valid_roc_auc"),
        "best_valid_f1": job.summary.get("best_valid_f1"),
        "peak_vram_mb": job.summary.get("peak_vram_mb"),
        "collapse_detected": job.summary.get("collapse_detected"),
    }


def write_status(output_root: Path, jobs: list[Job], event: dict | None = None) -> None:
    data = {
        "updated_at": now(),
        "counts": {s: sum(1 for j in jobs if j.status == s) for s in ["PENDING", "RUNNING", "OK", "COLLAPSED", "FAILED", "SKIPPED"]},
        "jobs": [payload(job) for job in jobs],
    }
    write_json(output_root / STATUS_JSON, data)
    if event is not None:
        append_jsonl(output_root / STATUS_JSONL, {"time": now(), **event})


def completed(job: Job) -> bool:
    summary_path = job.run_dir / "summary.json"
    if not summary_path.exists():
        return False
    job.summary = read_json(summary_path)
    job.status = str(job.summary.get("status", "OK"))
    return job.status in {"OK", "COLLAPSED"}


def discover(configs_root: Path) -> list[Job]:
    return [build_job(path) for path in sorted(configs_root.glob("*.yaml"))]


def run_job(job: Job, args: argparse.Namespace) -> int:
    job.run_dir.mkdir(parents=True, exist_ok=True)
    cmd = [args.python, str(ROOT / "scripts" / "04_train.py"), "--config", str(job.config_path)]
    log_path = job.run_dir / "train.log"
    job.started_at = now()
    job.status = "RUNNING"
    write_json(job.run_dir / "status.json", payload(job) | {"cmd": cmd})
    write_status(args.output_root, [job], {"event": "start", "run_name": job.run_name, "cmd": cmd})
    print(f"START {job.run_name} log={log_path}", flush=True)
    with log_path.open("w") as log:
        process = subprocess.Popen(cmd, cwd=str(ROOT), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        job.return_code = process.wait()
    job.finished_at = now()
    summary_path = job.run_dir / "summary.json"
    if summary_path.exists():
        job.summary = read_json(summary_path)
    job.status = str(job.summary.get("status", "OK")) if job.return_code == 0 else "FAILED"
    write_json(job.run_dir / "status.json", payload(job) | {"cmd": cmd})
    write_status(args.output_root, [job], {"event": "finish", "run_name": job.run_name, "status": job.status, "return_code": job.return_code})
    print(f"DONE {job.run_name} status={job.status} return_code={job.return_code}", flush=True)
    return int(job.return_code or 0)


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    if args.dry_run:
        jobs = [build_job(args.config)] if args.config else discover(args.configs_root)
        for job in jobs:
            completed(job)
        write_status(args.output_root, jobs, {"event": "dry_run", "jobs": len(jobs)})
        print(f"Checkpoint full dry-run: jobs={len(jobs)}")
        for job in jobs:
            print(f"DRY {job.status:10} bs={job.batch_size:<3} expected_vram_mb={job.expected_vram_mb:<5} run={job.run_name} config={job.config_path}")
        return
    if args.config is None:
        raise SystemExit("--config is required unless --dry-run is used")
    job = build_job(args.config)
    if completed(job) and not args.rerun_completed:
        write_status(args.output_root, [job], {"event": "skip_completed", "run_name": job.run_name})
        print(f"SKIP completed run {job.run_name}; pass --rerun-completed to run again")
        return
    raise SystemExit(run_job(job, args))


if __name__ == "__main__":
    main()

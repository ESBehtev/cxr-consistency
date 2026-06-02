from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import argparse
import json
import os
import subprocess
import time

import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_ROOT = ROOT / "configs" / "finalists"
DEFAULT_OUTPUT_ROOT = ROOT / "experiments" / "finalists"


@dataclass
class Job:
    config_path: Path
    run_name: str
    family: str
    model_type: str
    run_dir: Path
    expected_vram_mb: int
    status: str = "PENDING"
    process: subprocess.Popen | None = None
    log_handle: object | None = None
    started_at: float | None = None
    finished_at: float | None = None
    return_code: int | None = None
    summary: dict = field(default_factory=dict)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Queue-based launcher for final-stage A6000 runs.")
    parser.add_argument("configs", nargs="*", type=Path, help="Optional explicit finalist config files. Defaults to configs/finalists/*.yaml")
    parser.add_argument("--configs-root", type=Path, default=DEFAULT_CONFIG_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--max-parallel-jobs", type=int, default=4)
    parser.add_argument("--vram-reserve-mb", type=int, default=6000)
    parser.add_argument("--min-free-vram-mb", type=int, default=8000)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--python", type=str, default=str(ROOT / ".venv" / "bin" / "python"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--rerun-completed", action="store_true", help="Allow launching runs that already have summary.json.")
    parser.add_argument("--rerun-incomplete", action="store_true", help="Allow launching run dirs with partial files but no summary.json.")
    return parser.parse_args()


def now() -> float:
    return time.time()


def read_yaml(path: Path) -> dict:
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


def pid_alive(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        os.kill(int(pid), 0)
        return True
    except OSError:
        return False


def infer_family(config: dict) -> str:
    family = config.get("run_family")
    if family:
        return str(family)
    encoder = str(config.get("image_encoder_name") or "").lower()
    if encoder.startswith("convnext"):
        return "convnext"
    if encoder.startswith("deit"):
        return "deit"
    if encoder.startswith("vit"):
        return "vit"
    return "unknown"


def discover_configs(args: argparse.Namespace) -> list[Path]:
    paths = args.configs or sorted(args.configs_root.glob("*.yaml"))
    return [path.resolve() for path in paths]


def build_jobs(config_paths: list[Path]) -> list[Job]:
    jobs = []
    for config_path in config_paths:
        cfg = read_yaml(config_path)
        run_name = str(cfg.get("run_name") or config_path.stem)
        run_dir = Path(cfg.get("experiment_dir") or DEFAULT_OUTPUT_ROOT / run_name)
        if not run_dir.is_absolute():
            run_dir = ROOT / run_dir
        resources = cfg.get("resources", {}) or {}
        jobs.append(
            Job(
                config_path=config_path,
                run_name=run_name,
                family=infer_family(cfg),
                model_type=str(cfg.get("model_type") or cfg.get("image_encoder_name") or "unknown"),
                run_dir=run_dir,
                expected_vram_mb=int(resources.get("expected_vram_mb", 12000)),
            )
        )
    return jobs


def gpu_memory_mb() -> tuple[int | None, int | None]:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free,memory.total", "--format=csv,noheader,nounits"],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None, None
    pairs = []
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",") if part.strip()]
        if len(parts) == 2:
            pairs.append((int(parts[0]), int(parts[1])))
    return max(pairs, key=lambda item: item[0]) if pairs else (None, None)


def expected_running_vram(running: list[Job]) -> int:
    return sum(job.expected_vram_mb for job in running)


def runtime_min(job: Job) -> float | None:
    if job.started_at is None:
        return None
    return ((job.finished_at or now()) - job.started_at) / 60.0


def job_payload(job: Job) -> dict:
    return {
        "run_name": job.run_name,
        "family": job.family,
        "model_type": job.model_type,
        "config_path": str(job.config_path),
        "run_dir": str(job.run_dir),
        "expected_vram_mb": job.expected_vram_mb,
        "pid": job.process.pid if job.process else None,
        "status": job.status,
        "started_at": job.started_at,
        "finished_at": job.finished_at,
        "runtime_min": runtime_min(job),
        "return_code": job.return_code,
        "best_valid_roc_auc": job.summary.get("best_valid_roc_auc"),
        "best_valid_f1": job.summary.get("best_valid_f1"),
        "peak_vram_mb": job.summary.get("peak_vram_mb"),
        "collapse_detected": job.summary.get("collapse_detected"),
    }


def write_status(jobs: list[Job], output_root: Path, event: dict | None = None) -> None:
    payload = {
        "updated_at": now(),
        "counts": {status: sum(1 for job in jobs if job.status == status) for status in ["PENDING", "RUNNING", "OK", "COLLAPSED", "FAILED", "SKIPPED"]},
        "jobs": [job_payload(job) for job in jobs],
    }
    write_json(output_root / "finalists_status.json", payload)
    if event is not None:
        append_jsonl(output_root / "finalists_status.jsonl", {"time": now(), **event})


def preflight_job(job: Job, args: argparse.Namespace) -> None:
    summary_path = job.run_dir / "summary.json"
    status = read_json(job.run_dir / "status.json")
    if summary_path.exists() and not args.rerun_completed:
        job.summary = read_json(summary_path)
        job.status = str(job.summary.get("status", "OK"))
        return
    pid = status.get("pid")
    if pid_alive(pid):
        job.status = "SKIPPED"
        job.summary = {"skip_reason": f"existing live pid {pid}"}
        return
    has_partial = any((job.run_dir / name).exists() for name in ["metrics.jsonl", "train.log", "config_snapshot.yaml"])
    if has_partial and not summary_path.exists() and not args.rerun_incomplete:
        job.status = "SKIPPED"
        job.summary = {"skip_reason": "partial run dir exists; pass --rerun-incomplete to overwrite train.log"}


def can_launch(job: Job, running: list[Job], args: argparse.Namespace, free_mb: int | None, total_mb: int | None) -> bool:
    if len(running) >= args.max_parallel_jobs:
        return False
    projected = expected_running_vram(running) + job.expected_vram_mb
    if total_mb is not None and projected > total_mb - args.vram_reserve_mb:
        return False
    if free_mb is not None and free_mb < max(args.min_free_vram_mb, job.expected_vram_mb + args.vram_reserve_mb):
        return False
    return True


def launch_job(job: Job, args: argparse.Namespace, jobs: list[Job]) -> None:
    job.run_dir.mkdir(parents=True, exist_ok=True)
    log_path = job.run_dir / "train.log"
    job.log_handle = log_path.open("w")
    cmd = [args.python, str(ROOT / "scripts" / "04_train.py"), "--config", str(job.config_path)]
    job.process = subprocess.Popen(cmd, cwd=str(ROOT), stdout=job.log_handle, stderr=subprocess.STDOUT, text=True)
    job.started_at = now()
    job.status = "RUNNING"
    write_json(job.run_dir / "status.json", job_payload(job) | {"cmd": cmd})
    write_status(jobs, args.output_root, {"event": "start", "run_name": job.run_name, "pid": job.process.pid})
    print(f"START {job.run_name} pid={job.process.pid} log={log_path}", flush=True)


def finish_job(job: Job, args: argparse.Namespace, jobs: list[Job]) -> None:
    assert job.process is not None
    job.return_code = job.process.returncode
    job.finished_at = now()
    if job.log_handle:
        job.log_handle.close()
    summary_path = job.run_dir / "summary.json"
    if summary_path.exists():
        job.summary = read_json(summary_path)
    job.status = str(job.summary.get("status", "OK")) if job.return_code == 0 else "FAILED"
    write_json(job.run_dir / "status.json", job_payload(job))
    write_status(jobs, args.output_root, {"event": "finish", "run_name": job.run_name, "status": job.status, "return_code": job.return_code})
    print(f"DONE  {job.run_name} status={job.status} return_code={job.return_code}", flush=True)


def print_table(jobs: list[Job], free_mb: int | None) -> None:
    print(f"\nfree_vram_mb={free_mb if free_mb is not None else 'unknown'}")
    print("run_name                         family    model        status     pid      min   vram")
    for job in jobs:
        pid = job.process.pid if job.process else "-"
        print(f"{job.run_name[:32]:32} {job.family[:9]:9} {job.model_type[:12]:12} {job.status[:10]:10} {str(pid):8} {(runtime_min(job) or 0):5.1f} {job.expected_vram_mb:6}")


def main() -> None:
    args = parse_args()
    jobs = build_jobs(discover_configs(args))
    if not jobs:
        raise SystemExit("No finalist configs found")
    for job in jobs:
        preflight_job(job, args)
    if args.dry_run:
        print(f"Finalist queue dry-run: jobs={len(jobs)} max_parallel_jobs={args.max_parallel_jobs}")
        for job in jobs:
            print(f"DRY {job.status:10} {job.family:8} {job.model_type:12} {job.run_name:32} expected_vram_mb={job.expected_vram_mb} config={job.config_path}")
        write_status(jobs, args.output_root, {"event": "dry_run", "jobs": len(jobs)})
        return
    write_status(jobs, args.output_root, {"event": "queue_start", "jobs": len(jobs)})
    pending = [job for job in jobs if job.status == "PENDING"]
    running: list[Job] = []
    while pending or running:
        for job in running[:]:
            if job.process and job.process.poll() is not None:
                finish_job(job, args, jobs)
                running.remove(job)
        free_mb, total_mb = gpu_memory_mb()
        launched = True
        while pending and launched:
            launched = False
            for job in pending[:]:
                free_mb, total_mb = gpu_memory_mb()
                if can_launch(job, running, args, free_mb, total_mb):
                    pending.remove(job)
                    launch_job(job, args, jobs)
                    running.append(job)
                    launched = True
                    time.sleep(2.0)
                    break
        print_table(jobs, free_mb)
        if pending or running:
            time.sleep(args.poll_seconds)
    write_status(jobs, args.output_root, {"event": "queue_done", "jobs": len(jobs)})
    print("Finalist queue finished", flush=True)


if __name__ == "__main__":
    main()

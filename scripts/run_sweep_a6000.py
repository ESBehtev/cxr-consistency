from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import argparse
import json
import subprocess
import time

import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_ROOT = ROOT / "configs" / "sweeps"
DEFAULT_OUTPUT_ROOT = ROOT / "experiments" / "sweeps"
TERMINAL_STATUSES = {"OK", "COLLAPSED", "FAILED"}


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
    parser = argparse.ArgumentParser(description="Queue-based local sweep launcher for RTX A6000.")
    parser.add_argument("configs", nargs="*", type=Path, help="Optional explicit config files. Defaults to configs/sweeps/**/*.yaml")
    parser.add_argument("--configs-root", type=Path, default=DEFAULT_CONFIG_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--max-parallel-jobs", type=int, default=2)
    parser.add_argument("--vram-reserve-mb", type=int, default=6000)
    parser.add_argument("--min-free-vram-mb", type=int, default=8000)
    parser.add_argument("--poll-seconds", type=float, default=20.0)
    parser.add_argument("--python", type=str, default=str(ROOT / ".venv" / "bin" / "python"))
    parser.add_argument("--family", choices=["convnext", "vit", "deit"], action="append", help="Restrict to one or more families.")
    parser.add_argument("--dry-run", action="store_true", help="Print planned queue without launching training.")
    return parser.parse_args()


def now() -> float:
    return time.time()


def read_yaml(path: Path) -> dict:
    with path.open() as f:
        return yaml.safe_load(f)


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


def append_jsonl(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        json.dump(payload, f, sort_keys=True)
        f.write("\n")


def discover_configs(args: argparse.Namespace) -> list[Path]:
    if args.configs:
        configs = args.configs
    else:
        configs = sorted(args.configs_root.glob("**/*.yaml"))

    out = []
    for path in configs:
        path = path.resolve()
        if args.family and path.parent.name not in set(args.family):
            continue
        out.append(path)
    return out


def infer_family(config_path: Path, config: dict) -> str:
    if config.get("run_family"):
        return str(config["run_family"])
    parent = config_path.parent.name.lower()
    if parent in {"convnext", "vit", "deit"}:
        return parent
    name = str(config.get("image_encoder_name", "")).lower()
    if name.startswith("convnext"):
        return "convnext"
    if name.startswith("deit"):
        return "deit"
    if name.startswith("vit"):
        return "vit"
    return "unknown"


def build_jobs(config_paths: list[Path], output_root: Path) -> list[Job]:
    jobs = []
    for config_path in config_paths:
        config = read_yaml(config_path)
        run_name = str(config.get("run_name") or config_path.stem)
        family = infer_family(config_path, config)
        model_type = str(config.get("model_type") or config.get("image_encoder_name") or "unknown")
        run_dir = Path(config.get("experiment_dir") or output_root / run_name)
        if not run_dir.is_absolute():
            run_dir = ROOT / run_dir
        resources = config.get("resources", {}) or {}
        expected_vram_mb = int(resources.get("expected_vram_mb", config.get("expected_vram_mb", 12000)))
        jobs.append(
            Job(
                config_path=config_path,
                run_name=run_name,
                family=family,
                model_type=model_type,
                run_dir=run_dir,
                expected_vram_mb=expected_vram_mb,
            )
        )
    return jobs


def gpu_memory_mb() -> tuple[int | None, int | None]:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.free,memory.total",
                "--format=csv,noheader,nounits",
            ],
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
    if not pairs:
        return None, None
    return max(pairs, key=lambda item: item[0])


def free_vram_mb() -> int | None:
    free_mb, _ = gpu_memory_mb()
    return free_mb


def total_expected_vram(running: list[Job]) -> int:
    return sum(job.expected_vram_mb for job in running)


def runtime_min(job: Job) -> float | None:
    if job.started_at is None:
        return None
    end = job.finished_at or now()
    return (end - job.started_at) / 60.0


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
    output_root.mkdir(parents=True, exist_ok=True)
    payload = {
        "updated_at": now(),
        "counts": {status: sum(1 for job in jobs if job.status == status) for status in ["PENDING", "RUNNING", "OK", "COLLAPSED", "FAILED"]},
        "jobs": [job_payload(job) for job in jobs],
    }
    write_json(output_root / "sweep_status.json", payload)
    if event is not None:
        append_jsonl(output_root / "sweep_status.jsonl", {"time": now(), **event})


def print_status_table(jobs: list[Job], free_mb: int | None) -> None:
    counts = {status: sum(1 for job in jobs if job.status == status) for status in ["PENDING", "RUNNING", "OK", "COLLAPSED", "FAILED"]}
    print(
        "\nSTATUS "
        + " ".join(f"{k}={v}" for k, v in counts.items())
        + f" free_vram_mb={free_mb if free_mb is not None else 'unknown'}",
        flush=True,
    )
    print("run_name                     family    model              status     pid      min   auc     f1      vram", flush=True)
    for job in jobs:
        payload = job_payload(job)
        print(
            f"{job.run_name[:28]:28} {job.family[:9]:9} {job.model_type[:18]:18} "
            f"{job.status[:10]:10} {str(payload['pid'] or '-'):8} "
            f"{(payload['runtime_min'] or 0):5.1f} "
            f"{payload['best_valid_roc_auc'] if payload['best_valid_roc_auc'] is not None else '-':>7} "
            f"{payload['best_valid_f1'] if payload['best_valid_f1'] is not None else '-':>7} "
            f"{payload['peak_vram_mb'] if payload['peak_vram_mb'] is not None else '-':>7}",
            flush=True,
        )


def can_launch(job: Job, running: list[Job], args: argparse.Namespace, free_mb: int | None) -> bool:
    if len(running) >= args.max_parallel_jobs:
        return False
    _, total_mb = gpu_memory_mb()
    projected_expected = total_expected_vram(running) + job.expected_vram_mb
    if total_mb is not None and projected_expected > total_mb - args.vram_reserve_mb:
        return False
    if free_mb is None:
        return True
    if free_mb < args.min_free_vram_mb:
        return False
    return free_mb >= job.expected_vram_mb + args.vram_reserve_mb


def launch_job(job: Job, args: argparse.Namespace, jobs: list[Job]) -> None:
    job.run_dir.mkdir(parents=True, exist_ok=True)
    log_path = job.run_dir / "train.log"
    job.log_handle = log_path.open("w")
    cmd = [args.python, str(ROOT / "scripts" / "04_train.py"), "--config", str(job.config_path)]
    job.process = subprocess.Popen(cmd, cwd=str(ROOT), stdout=job.log_handle, stderr=subprocess.STDOUT, text=True)
    job.started_at = now()
    job.status = "RUNNING"
    write_json(job.run_dir / "status.json", job_payload(job) | {"cmd": cmd})
    write_status(jobs, args.output_root, {"event": "start", "run_name": job.run_name, "pid": job.process.pid, "cmd": cmd})
    print(f"START {job.run_name} pid={job.process.pid} expected_vram_mb={job.expected_vram_mb} log={log_path}", flush=True)


def finish_job(job: Job, jobs: list[Job], args: argparse.Namespace) -> None:
    assert job.process is not None
    job.return_code = job.process.returncode
    job.finished_at = now()
    if job.log_handle:
        job.log_handle.close()
    summary_path = job.run_dir / "summary.json"
    if summary_path.exists():
        job.summary = json.loads(summary_path.read_text())
    if job.return_code == 0:
        job.status = str(job.summary.get("status", "OK"))
    else:
        job.status = "FAILED"
    write_json(job.run_dir / "status.json", job_payload(job))
    write_status(jobs, args.output_root, {"event": "finish", "run_name": job.run_name, "status": job.status, "return_code": job.return_code})
    print(f"DONE  {job.run_name} status={job.status} return_code={job.return_code}", flush=True)


def main() -> None:
    args = parse_args()
    config_paths = discover_configs(args)
    jobs = build_jobs(config_paths, args.output_root)
    pending = jobs[:]
    running: list[Job] = []

    if not jobs:
        raise SystemExit("No sweep configs found")

    print(f"Sweep queue: configs={len(jobs)} max_parallel_jobs={args.max_parallel_jobs} output_root={args.output_root}", flush=True)
    if args.dry_run:
        for job in jobs:
            print(f"DRY {job.family:8} {job.model_type:16} {job.run_name:28} expected_vram_mb={job.expected_vram_mb} config={job.config_path}")
        write_status(jobs, args.output_root, {"event": "dry_run", "jobs": len(jobs)})
        return

    write_status(jobs, args.output_root, {"event": "queue_start", "jobs": len(jobs)})

    while pending or running:
        for job in running[:]:
            if job.process and job.process.poll() is not None:
                finish_job(job, jobs, args)
                running.remove(job)

        free_mb = free_vram_mb()
        launched = True
        while pending and launched:
            launched = False
            for job in pending[:]:
                free_mb = free_vram_mb()
                if can_launch(job, running, args, free_mb):
                    pending.remove(job)
                    launch_job(job, args, jobs)
                    running.append(job)
                    launched = True
                    time.sleep(2.0)
                    break

        print_status_table(jobs, free_mb)
        if pending or running:
            time.sleep(args.poll_seconds)

    write_status(jobs, args.output_root, {"event": "queue_done", "jobs": len(jobs)})
    failed = [job for job in jobs if job.status == "FAILED"]
    collapsed = [job for job in jobs if job.status == "COLLAPSED"]
    print(f"Sweep finished: total={len(jobs)} failed={len(failed)} collapsed={len(collapsed)}", flush=True)


if __name__ == "__main__":
    main()

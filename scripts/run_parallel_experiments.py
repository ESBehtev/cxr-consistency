from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import argparse
import json
import subprocess
import sys
import time

import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "experiments" / "parallel_a6000"


@dataclass
class Job:
    config_path: Path
    run_name: str
    run_dir: Path
    process: subprocess.Popen | None = None
    log_handle: object | None = None
    started_at: float | None = None
    finished_at: float | None = None
    status: str = "PENDING"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("configs", nargs="+", type=Path)
    parser.add_argument("--max-parallel", type=int, default=2)
    parser.add_argument("--min-free-vram-mb", type=int, default=12000)
    parser.add_argument("--poll-seconds", type=float, default=10.0)
    parser.add_argument("--python", type=str, default=str(ROOT / ".venv" / "bin" / "python"))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def read_config(path: Path) -> dict:
    with path.open() as f:
        return yaml.safe_load(f)


def free_vram_mb() -> int | None:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.free",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None

    values = [int(line.strip()) for line in result.stdout.splitlines() if line.strip()]
    return max(values) if values else None


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


def append_status(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        json.dump(payload, f, sort_keys=True)
        f.write("\n")


def update_job_status(job: Job, status_path: Path, extra: dict | None = None) -> None:
    payload = {
        "run_name": job.run_name,
        "config": str(job.config_path),
        "run_dir": str(job.run_dir),
        "pid": job.process.pid if job.process else None,
        "status": job.status,
        "started_at": job.started_at,
        "finished_at": job.finished_at,
        "runtime_min": ((job.finished_at or time.time()) - job.started_at) / 60.0 if job.started_at else None,
    }
    if extra:
        payload.update(extra)
    write_json(job.run_dir / "status.json", payload)
    append_status(status_path, payload)


def build_jobs(config_paths: list[Path]) -> list[Job]:
    jobs = []
    for config_path in config_paths:
        config_path = config_path.resolve()
        config = read_config(config_path)
        run_name = str(config.get("run_name") or config_path.stem)
        run_dir = Path(config.get("experiment_dir") or DEFAULT_OUTPUT_DIR / run_name)
        if not run_dir.is_absolute():
            run_dir = ROOT / run_dir
        jobs.append(Job(config_path=config_path, run_name=run_name, run_dir=run_dir))
    return jobs


def launch_job(job: Job, python_bin: str, status_path: Path) -> None:
    job.run_dir.mkdir(parents=True, exist_ok=True)
    log_path = job.run_dir / "train.log"
    job.log_handle = log_path.open("w")
    cmd = [python_bin, str(ROOT / "scripts" / "04_train.py"), "--config", str(job.config_path)]
    job.process = subprocess.Popen(
        cmd,
        cwd=str(ROOT),
        stdout=job.log_handle,
        stderr=subprocess.STDOUT,
        text=True,
    )
    job.started_at = time.time()
    job.status = "RUNNING"
    update_job_status(job, status_path, {"cmd": cmd})
    print(f"START {job.run_name} pid={job.process.pid} log={log_path}", flush=True)


def finish_job(job: Job, status_path: Path) -> None:
    assert job.process is not None
    return_code = job.process.returncode
    job.finished_at = time.time()
    if job.log_handle:
        job.log_handle.close()
    if return_code == 0:
        summary_path = job.run_dir / "summary.json"
        if summary_path.exists():
            summary = json.loads(summary_path.read_text())
            job.status = str(summary.get("status", "OK"))
        else:
            job.status = "OK"
    else:
        job.status = "FAILED"
    update_job_status(job, status_path, {"return_code": return_code})
    print(f"DONE  {job.run_name} status={job.status} return_code={return_code}", flush=True)


def main() -> None:
    args = parse_args()
    status_path = args.output_dir / "launcher_status.jsonl"
    jobs = build_jobs(args.configs)
    pending = jobs[:]
    running: list[Job] = []
    completed: list[Job] = []

    print(
        f"Launcher: jobs={len(jobs)} max_parallel={args.max_parallel} "
        f"min_free_vram_mb={args.min_free_vram_mb}",
        flush=True,
    )

    while pending or running:
        for job in running[:]:
            if job.process and job.process.poll() is not None:
                finish_job(job, status_path)
                running.remove(job)
                completed.append(job)

        while pending and len(running) < args.max_parallel:
            free_mb = free_vram_mb()
            if free_mb is not None and free_mb < args.min_free_vram_mb:
                print(f"WAIT free_vram_mb={free_mb} < {args.min_free_vram_mb}", flush=True)
                break
            job = pending.pop(0)
            launch_job(job, args.python, status_path)
            running.append(job)
            time.sleep(2.0)

        if pending or running:
            time.sleep(args.poll_seconds)

    failures = [job for job in completed if job.status not in {"OK", "COLLAPSED"}]
    print(f"Launcher finished: completed={len(completed)} failures={len(failures)}", flush=True)
    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()

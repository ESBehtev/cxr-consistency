from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import argparse
import json
import subprocess
import time

try:
    import yaml
except ModuleNotFoundError:  # pragma: no cover - used only in minimal shells
    yaml = None

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_ROOT = ROOT / "configs" / "final_full"
DEFAULT_OUTPUT_ROOT = ROOT / "experiments" / "final_full"

@dataclass
class Job:
    config_path: Path
    run_name: str
    family: str
    image_encoder: str
    expected_vram_mb: int
    run_dir: Path
    status: str = "PENDING"
    process: subprocess.Popen | None = None
    log_handle: object | None = None
    started_at: float | None = None
    finished_at: float | None = None
    return_code: int | None = None
    summary: dict = field(default_factory=dict)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="VRAM-aware launcher for final full CXR consistency runs.")
    parser.add_argument("configs", nargs="*", type=Path)
    parser.add_argument("--configs-root", type=Path, default=DEFAULT_CONFIG_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--max-parallel-jobs", type=int, default=3)
    parser.add_argument("--vram-reserve-mb", type=int, default=6000)
    parser.add_argument("--min-free-vram-mb", type=int, default=8000)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--python", type=str, default=str(ROOT / ".venv" / "bin" / "python"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--rerun-completed", action="store_true")
    return parser.parse_args()

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
    if yaml is not None:
        with path.open() as f:
            return yaml.safe_load(f) or {}
    return read_simple_yaml(path)

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

def now() -> float:
    return time.time()

def discover_configs(args: argparse.Namespace) -> list[Path]:
    return [p.resolve() for p in (args.configs or sorted(args.configs_root.glob("*.yaml")))]

def infer_family(cfg: dict) -> str:
    family = cfg.get("run_family")
    if family:
        return str(family)
    image = str(cfg.get("image_encoder_name") or "").lower()
    if image.startswith("convnext"):
        return "convnext"
    if image.startswith("deit"):
        return "deit"
    if image.startswith("vit"):
        return "vit"
    return "unknown"

def build_jobs(paths: list[Path]) -> list[Job]:
    jobs = []
    for path in paths:
        cfg = read_yaml(path)
        run_name = str(cfg.get("run_name") or path.stem)
        run_dir = Path(cfg.get("experiment_dir") or DEFAULT_OUTPUT_ROOT / run_name)
        if not run_dir.is_absolute():
            run_dir = ROOT / run_dir
        resources = cfg.get("resources", {}) or {}
        jobs.append(Job(
            config_path=path,
            run_name=run_name,
            family=infer_family(cfg),
            image_encoder=str(cfg.get("image_encoder_name") or "unknown"),
            expected_vram_mb=int(resources.get("expected_vram_mb", 12000)),
            run_dir=run_dir,
        ))
    return jobs

def gpu_memory_mb() -> tuple[int | None, int | None]:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free,memory.total", "--format=csv,noheader,nounits"],
            check=True, capture_output=True, text=True,
        )
    except Exception:
        return None, None
    pairs = []
    for line in result.stdout.splitlines():
        parts = [p.strip() for p in line.split(",") if p.strip()]
        if len(parts) == 2:
            pairs.append((int(parts[0]), int(parts[1])))
    return max(pairs, key=lambda x: x[0]) if pairs else (None, None)

def runtime_min(job: Job) -> float | None:
    if job.started_at is None:
        return None
    return ((job.finished_at or now()) - job.started_at) / 60.0

def payload(job: Job) -> dict:
    return {
        "run_name": job.run_name,
        "family": job.family,
        "image_encoder": job.image_encoder,
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
    data = {
        "updated_at": now(),
        "counts": {s: sum(1 for j in jobs if j.status == s) for s in ["PENDING", "RUNNING", "OK", "COLLAPSED", "FAILED", "SKIPPED"]},
        "jobs": [payload(j) for j in jobs],
    }
    write_json(output_root / "final_full_status.json", data)
    if event is not None:
        append_jsonl(output_root / "final_full_status.jsonl", {"time": now(), **event})

def preflight(job: Job, rerun_completed: bool) -> None:
    summary = job.run_dir / "summary.json"
    if summary.exists() and not rerun_completed:
        job.summary = read_json(summary)
        job.status = str(job.summary.get("status", "OK"))

def can_launch(job: Job, running: list[Job], args: argparse.Namespace, free_mb: int | None, total_mb: int | None) -> bool:
    if len(running) >= args.max_parallel_jobs:
        return False
    projected = sum(j.expected_vram_mb for j in running) + job.expected_vram_mb
    if total_mb is not None and projected > total_mb - args.vram_reserve_mb:
        return False
    if free_mb is not None and free_mb < max(args.min_free_vram_mb, job.expected_vram_mb + args.vram_reserve_mb):
        return False
    return True

def launch(job: Job, args: argparse.Namespace, jobs: list[Job]) -> None:
    job.run_dir.mkdir(parents=True, exist_ok=True)
    log_path = job.run_dir / "train.log"
    job.log_handle = log_path.open("w")
    cmd = [args.python, str(ROOT / "scripts" / "04_train.py"), "--config", str(job.config_path)]
    job.process = subprocess.Popen(cmd, cwd=str(ROOT), stdout=job.log_handle, stderr=subprocess.STDOUT, text=True)
    job.started_at = now()
    job.status = "RUNNING"
    write_json(job.run_dir / "status.json", payload(job) | {"cmd": cmd})
    write_status(jobs, args.output_root, {"event": "start", "run_name": job.run_name, "pid": job.process.pid})
    print(f"START {job.run_name} pid={job.process.pid} log={log_path}", flush=True)

def finish(job: Job, args: argparse.Namespace, jobs: list[Job]) -> None:
    assert job.process is not None
    job.return_code = job.process.returncode
    job.finished_at = now()
    if job.log_handle:
        job.log_handle.close()
    summary = job.run_dir / "summary.json"
    if summary.exists():
        job.summary = read_json(summary)
    job.status = str(job.summary.get("status", "OK")) if job.return_code == 0 else "FAILED"
    write_json(job.run_dir / "status.json", payload(job))
    write_status(jobs, args.output_root, {"event": "finish", "run_name": job.run_name, "status": job.status, "return_code": job.return_code})
    print(f"DONE {job.run_name} status={job.status} return_code={job.return_code}", flush=True)

def print_table(jobs: list[Job], free_mb: int | None) -> None:
    print(f"\nfree_vram_mb={free_mb if free_mb is not None else 'unknown'}")
    print("run_name                     family    image          status     pid      min   expected_vram")
    for job in jobs:
        pid = job.process.pid if job.process else "-"
        print(f"{job.run_name[:28]:28} {job.family[:9]:9} {job.image_encoder[:14]:14} {job.status[:10]:10} {str(pid):8} {(runtime_min(job) or 0):5.1f} {job.expected_vram_mb:6}")

def main() -> None:
    args = parse_args()
    jobs = build_jobs(discover_configs(args))
    if not jobs:
        raise SystemExit("No final full configs found")
    for job in jobs:
        preflight(job, args.rerun_completed)
    if args.dry_run:
        print(f"Final full dry-run: jobs={len(jobs)} max_parallel_jobs={args.max_parallel_jobs}")
        for job in jobs:
            print(f"DRY {job.status:10} {job.family:8} {job.image_encoder:14} run={job.run_name} expected_vram_mb={job.expected_vram_mb} config={job.config_path}")
        write_status(jobs, args.output_root, {"event": "dry_run", "jobs": len(jobs)})
        return
    write_status(jobs, args.output_root, {"event": "queue_start", "jobs": len(jobs)})
    pending = [j for j in jobs if j.status == "PENDING"]
    running: list[Job] = []
    while pending or running:
        for job in running[:]:
            if job.process and job.process.poll() is not None:
                finish(job, args, jobs)
                running.remove(job)
        free_mb, total_mb = gpu_memory_mb()
        launched = True
        while pending and launched:
            launched = False
            for job in pending[:]:
                free_mb, total_mb = gpu_memory_mb()
                if can_launch(job, running, args, free_mb, total_mb):
                    pending.remove(job)
                    launch(job, args, jobs)
                    running.append(job)
                    launched = True
                    time.sleep(2.0)
                    break
        print_table(jobs, free_mb)
        if pending or running:
            time.sleep(args.poll_seconds)
    write_status(jobs, args.output_root, {"event": "queue_done", "jobs": len(jobs)})
    print("Final full queue finished", flush=True)

if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Job:
    tool: str
    architecture: str
    scope: str
    lut: str
    timeout_seconds: int

    @property
    def tag(self) -> str:
        return f"{self.scope}_{self.architecture}_lut-{self.lut}"


def tail(path: Path, lines: int = 120) -> str:
    if not path.exists():
        return f"<log does not exist: {path}>"
    return "\n".join(path.read_text(errors="replace").splitlines()[-lines:])


def valid_manifest(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size == 0:
        return False
    try:
        payload = json.loads(path.read_text())
    except Exception:
        return False
    keys = (
        ("nsys_report", "sqlite", "sql_summary_json")
        if "nsys_report" in payload
        else ("ncu_report", "raw_json", "details_text")
    )
    return all(Path(payload[key]).is_file() and Path(payload[key]).stat().st_size > 0 for key in keys)


def run_job(
    job: Job,
    *,
    output_root: Path,
    environment: dict[str, str],
    global_started: float,
    total_budget_seconds: int,
    skip_existing: bool,
) -> str:
    outdir = output_root / job.tool
    outdir.mkdir(parents=True, exist_ok=True)
    manifest = outdir / f"{job.tag}_manifest.json"
    if skip_existing and valid_manifest(manifest):
        print(f"SKIP valid existing profile: {manifest}", flush=True)
        return "skipped"

    remaining = total_budget_seconds - int(time.monotonic() - global_started)
    if remaining <= 45:
        print(f"SKIP {job.tag}: global time budget exhausted", flush=True)
        return "budget_exhausted"
    allowed = min(job.timeout_seconds, remaining - 30)

    script = (
        "scripts/profile_nsys_v014.sh"
        if job.tool == "nsys"
        else "scripts/profile_ncu_v014.sh"
    )
    command = ["bash", script, job.architecture, job.scope, job.lut, str(outdir)]
    print("\n" + "=" * 78, flush=True)
    print(f"{job.tool.upper()} {job.scope}: {job.architecture}, LUT={job.lut}", flush=True)
    print("Command:", " ".join(command), flush=True)
    print(f"Hard per-job limit: {allowed / 60:.1f} minutes", flush=True)
    print("=" * 78, flush=True)

    process = subprocess.Popen(
        command,
        env=environment,
        start_new_session=True,
    )
    try:
        return_code = process.wait(timeout=allowed)
    except subprocess.TimeoutExpired:
        print(f"TIMEOUT: stopping {job.tag}", flush=True)
        os.killpg(process.pid, signal.SIGINT)
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
        return "timeout"

    if return_code != 0:
        log_path = outdir / f"{job.tag}.log"
        raise RuntimeError(
            f"{job.tool.upper()} job failed with code {return_code}: {job.tag}\n"
            f"Log: {log_path}\n\nLast log lines:\n{tail(log_path)}"
        )
    if not valid_manifest(manifest):
        raise RuntimeError(
            f"{job.tool.upper()} returned success but its manifest/artifacts are incomplete: {manifest}"
        )
    print(f"PASS: {manifest}", flush=True)
    return "completed"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Article-essential Nsight profile suite with a strict total time limit."
    )
    parser.add_argument("--output-root", type=Path, default=Path("reports/v0.14.2"))
    parser.add_argument("--model", default="facebook/opt-2.7b")
    parser.add_argument("--matrix-shape", default="16x2560x2560")
    parser.add_argument("--attention-seq", type=int, default=64)
    parser.add_argument("--total-minutes", type=int, default=55)
    parser.add_argument("--run-nsys", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--run-ncu", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--ncu-mode", choices=["essential", "full"], default="essential")
    args = parser.parse_args()

    if args.total_minutes <= 0 or args.total_minutes > 60:
        raise SystemExit("--total-minutes must be in 1..60 for the default Colab protocol")

    for script in (
        Path("scripts/profile_workload_v014.py"),
        Path("scripts/profile_nsys_v014.sh"),
        Path("scripts/profile_ncu_v014.sh"),
    ):
        if not script.is_file():
            raise SystemExit(f"Missing {script}; run this command from the repository root")

    jobs: list[Job] = []
    if args.run_nsys:
        jobs += [
            Job("nsys", "fp16", "attention", "none", 4 * 60),
            Job("nsys", "full_rns_int8", "attention", "two", 5 * 60),
            Job("nsys", "hybrid_rns_q16", "attention", "two", 5 * 60),
        ]
    if args.run_ncu:
        jobs += [
            Job("ncu", "full_rns_int8", "matrix", "two", 18 * 60),
            Job("ncu", "hybrid_rns_q16", "matrix", "two", 18 * 60),
        ]

    environment = os.environ.copy()
    environment.update(
        {
            "MODEL_ID": args.model,
            "MATRIX_SHAPE": args.matrix_shape,
            "ATTENTION_SEQ": str(min(args.attention_seq, 64)),
            "NSYS_WARMUP": "2",
            "NSYS_ITERATIONS": "3",
            "NCU_WARMUP": "2",
            "NCU_ITERATIONS": "1",
            "NCU_MODE": args.ncu_mode,
            "NCU_MAX_LAUNCHES": "4",
        }
    )

    started = time.monotonic()
    outcomes: list[dict[str, str]] = []
    for job in jobs:
        outcome = run_job(
            job,
            output_root=args.output_root,
            environment=environment,
            global_started=started,
            total_budget_seconds=args.total_minutes * 60,
            skip_existing=args.skip_existing,
        )
        outcomes.append({"job": job.tag, "tool": job.tool, "outcome": outcome})

    elapsed = (time.monotonic() - started) / 60.0
    summary = {
        "version": "0.14.2",
        "protocol": "article-essential-under-one-hour",
        "elapsed_minutes": elapsed,
        "total_budget_minutes": args.total_minutes,
        "ncu_mode": args.ncu_mode,
        "jobs": outcomes,
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_root / "minimal_nsight_summary_v0142.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print("\n" + json.dumps(summary, indent=2), flush=True)
    print(f"Summary: {summary_path}", flush=True)


if __name__ == "__main__":
    main()

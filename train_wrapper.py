#!/usr/bin/env python3
import argparse
import atexit
import os
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from uuid import uuid4


def _is_pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _terminal_title(title: str) -> None:
    sys.stdout.write(f"\033]0;{title}\007")
    sys.stdout.flush()


def _with_run_id(path_value: str, run_id: str) -> str:
    p = Path(path_value)
    stem = p.stem if p.stem else p.name
    suffix = p.suffix
    parent = p.parent
    return str(parent / f"{stem}_{run_id}{suffix}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Single-run lock + run_id aware launcher for sncp_ppo.train"
    )
    parser.add_argument("--save_path", type=str, required=True)
    parser.add_argument("--on_busy_save_path", choices=["abort", "suffix"], default="abort")
    parser.add_argument("--lockfile", type=str, default=".train_wrapper.lock")
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("train_args", nargs=argparse.REMAINDER,
                        help="Extra args passed to `python -m sncp_ppo.train`.")
    args = parser.parse_args()

    run_id = args.run_id or f"{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}_{uuid4().hex[:8]}"
    start_ts = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

    lockfile = Path(args.lockfile)
    lockfile.parent.mkdir(parents=True, exist_ok=True)

    if lockfile.exists():
        content = lockfile.read_text(encoding="utf-8", errors="ignore").strip().splitlines()
        pid = int(content[0]) if content and content[0].isdigit() else -1
        existing_run = content[1].strip() if len(content) > 1 else "unknown"
        if _is_pid_alive(pid):
            print(
                f"[ABORT] Active run lock detected: pid={pid}, run_id={existing_run}, lockfile={lockfile}",
                file=sys.stderr,
            )
            return 2
        print(f"[WARN] Stale lock detected (pid={pid}). Replacing lockfile: {lockfile}")

    requested_save = Path(args.save_path)
    active_marker = requested_save.with_suffix(requested_save.suffix + ".active")
    final_save = requested_save

    if active_marker.exists():
        marker_lines = active_marker.read_text(encoding="utf-8", errors="ignore").splitlines()
        marker_pid = int(marker_lines[0]) if marker_lines and marker_lines[0].isdigit() else -1
        marker_run = marker_lines[1] if len(marker_lines) > 1 else "unknown"
        if _is_pid_alive(marker_pid):
            msg = (
                f"[WARN] save_path already active: path={requested_save} pid={marker_pid} run_id={marker_run}"
            )
            if args.on_busy_save_path == "abort":
                print(msg + " -> abort", file=sys.stderr)
                return 3
            final_save = Path(_with_run_id(str(requested_save), run_id))
            active_marker = final_save.with_suffix(final_save.suffix + ".active")
            print(msg + f" -> suffix, using {final_save}")

    # attach run_id to outputs
    final_save = Path(_with_run_id(str(final_save), run_id))

    lockfile.write_text(f"{os.getpid()}\n{run_id}\n{start_ts}\n", encoding="utf-8")

    def _cleanup(*_):
        try:
            if active_marker.exists():
                active_marker.unlink()
        except Exception:
            pass
        try:
            if lockfile.exists():
                lockfile.unlink()
        except Exception:
            pass

    atexit.register(_cleanup)
    signal.signal(signal.SIGTERM, _cleanup)
    signal.signal(signal.SIGINT, _cleanup)

    active_marker.parent.mkdir(parents=True, exist_ok=True)
    active_marker.write_text(f"{os.getpid()}\n{run_id}\n{start_ts}\n", encoding="utf-8")

    title = f"sncp-train run_id={run_id} start={start_ts}"
    _terminal_title(title)
    print(f"[RUN] {title}")

    # Ensure logs CSV file includes run_id explicitly.
    default_csv = Path("logs") / f"training_{run_id}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.csv"

    passthrough = args.train_args
    if passthrough and passthrough[0] == "--":
        passthrough = passthrough[1:]

    cmd = [
        sys.executable,
        "-u",
        "-m",
        "sncp_ppo.train",
        "--save_path",
        str(final_save),
        "--run_id",
        run_id,
        "--csv_path",
        str(default_csv),
        *passthrough,
    ]

    print("[CMD]", " ".join(cmd))
    proc = subprocess.run(cmd)
    return proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())

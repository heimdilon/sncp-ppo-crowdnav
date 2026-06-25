"""Generate or execute the paired v37 C0/C1 probe matrix.

The default mode only prints the six Colab commands. ``--mode run`` trains and
evaluates them sequentially, preserving episode-level shared-case artifacts in
``eval_v37_probe`` for the preregistered analyzer.
"""

from __future__ import annotations

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import argparse
import csv
import json
import math
import shlex
import shutil
import subprocess
from dataclasses import asdict
from pathlib import Path

import torch

from sncp_ppo.eval_report import evaluate_density
from sncp_ppo.training_log_report import analyze_training_log
from sncp_ppo.v37_probe import ARMS, DENSITIES, TRAIN_SEEDS, build_training_command


def _load_state(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _training_diagnostics(log_path: Path | None) -> dict:
    if log_path is None or not log_path.exists():
        return {"finite": False, "collapse_delta": None, "training_log": None}
    summary = analyze_training_log(log_path, collapse_threshold=0.05)
    required = ("entropy", "approx_kl", "return_rms_std")
    seen = {key: False for key in required}
    finite = True
    with log_path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            for key in required:
                value = row.get(key, "")
                if value == "":
                    continue
                seen[key] = True
                finite = finite and math.isfinite(float(value))
    return {
        "finite": finite and all(seen.values()),
        "collapse_delta": summary.collapse_delta,
        "training_log": str(log_path),
    }


def _checkpoint_for(save_path: Path) -> Path:
    if save_path.exists():
        return save_path
    final = save_path.with_name(save_path.stem + "_final.pt")
    if final.exists():
        return final
    raise FileNotFoundError(f"probe checkpoint not found: {save_path} or {final}")


def _evaluate_run(*, arm: str, train_seed: int, checkpoint: Path,
                  output_dir: Path, eval_episodes: int, training_log: Path | None) -> Path:
    state = _load_state(checkpoint)
    gate = state.get("hh_gate")
    payload = {
        "arm": arm,
        "train_seed": train_seed,
        "checkpoint": str(checkpoint),
        "hh_gate": float(gate.item()) if gate is not None else None,
        "diagnostics": _training_diagnostics(training_log),
        "episodes": [],
    }
    output_path = output_dir / f"{arm}_s{train_seed}_episodes.json"
    for density in DENSITIES:
        episode_seed = 100_000 + density * 1_000
        results = evaluate_density(
            checkpoint_path=checkpoint,
            num_humans=density,
            scenario="paper_challenging",
            n_episodes=eval_episodes,
            seed=episode_seed,
            robot_vpref=1.0,
            human_vpref_override=1.0,
            max_time=None,
            human_goal_noise=0.0,
        )
        for index, result in enumerate(results):
            payload["episodes"].append({
                "case_id": f"n{density}_seed{episode_seed + index}",
                "density": density,
                "episode_seed": episode_seed + index,
                **asdict(result),
            })
        output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        success = sum(result.success for result in results) / len(results)
        print(f"{arm} seed={train_seed} N={density}: success={success:.1%}", flush=True)
    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("commands", "run", "evaluate"), default="commands")
    parser.add_argument("--base_checkpoint", type=Path, default=Path("sncp_ppo_v34.pt"))
    parser.add_argument("--output_dir", type=Path, default=Path("eval_v37_probe"))
    parser.add_argument("--python", default=_sys.executable)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(TRAIN_SEEDS))
    parser.add_argument("--arms", choices=ARMS, nargs="+", default=list(ARMS))
    parser.add_argument("--eval_episodes", type=int, default=100)
    parser.add_argument("--total_steps", type=int, default=300_000)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    jobs = []
    for seed in args.seeds:
        for arm in args.arms:
            save_path = Path("checkpoints") / f"sncp_ppo_v37_probe_{arm}_s{seed}.pt"
            command = build_training_command(
                arm,
                seed,
                base_checkpoint=args.base_checkpoint,
                save_path=save_path,
                python_executable=args.python,
                total_steps=args.total_steps,
            )
            jobs.append((arm, seed, save_path, command))
            if args.mode == "commands":
                print(shlex.join(command))
    if args.mode == "commands":
        print(
            f"\nAfter all six runs: {shlex.join([args.python, 'scratch/_analyze_v37_probe.py'])}"
        )
        return 0

    if not args.base_checkpoint.exists():
        raise FileNotFoundError(f"base checkpoint not found: {args.base_checkpoint}")

    for arm, seed, save_path, command in jobs:
        named_log = args.output_dir / f"{arm}_s{seed}_training.csv"
        if args.mode == "run":
            before = set(Path("logs").glob("training_*.csv")) if Path("logs").exists() else set()
            print(f"\nRUN {arm} seed={seed}: {shlex.join(command)}", flush=True)
            subprocess.run(command, check=True)
            after = set(Path("logs").glob("training_*.csv"))
            created = sorted(after - before, key=lambda path: path.stat().st_mtime)
            if not created:
                raise RuntimeError(f"training log not found for {arm} seed={seed}")
            shutil.copy2(created[-1], named_log)
        checkpoint = _checkpoint_for(save_path)
        log_path = named_log if named_log.exists() else None
        _evaluate_run(
            arm=arm,
            train_seed=seed,
            checkpoint=checkpoint,
            output_dir=args.output_dir,
            eval_episodes=args.eval_episodes,
            training_log=log_path,
        )
    print(f"Probe artifacts ready: {args.output_dir}")
    print(f"Analyze: {args.python} scratch/_analyze_v37_probe.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

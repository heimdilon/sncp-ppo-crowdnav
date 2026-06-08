"""One-command v17 post-run review pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from sncp_ppo.colab_artifacts import stage_colab_artifacts
from sncp_ppo.post_run_pipeline import find_latest_training_csv, run_v16_post_eval
from sncp_ppo.v18_decision import (
    select_v18_candidate,
    write_v18_decision_json,
    write_v18_decision_report,
)
from sncp_ppo.v18_gate import verify_v18_decision_ready, write_v18_gate_report


@dataclass(frozen=True)
class V17ReviewResult:
    status: str
    checkpoint_path: Path
    training_csv: Path
    output_dir: Path
    post_eval_status: str
    branch_id: str | None
    single_variable: str | None
    decision_report: Path
    gate_report: Path


PostEvalRunner = Callable[..., object]
DecisionSelector = Callable[..., object]
DecisionWriter = Callable[..., None]
GateVerifier = Callable[..., object]
GateWriter = Callable[..., None]


def run_v17_review(
    *,
    staging_dir: str | Path = "colabout",
    repo_root: str | Path = ".",
    stage_artifacts: bool = False,
    overwrite: bool = False,
    checkpoint_path: str | Path | None = None,
    training_csv: str | Path | None = None,
    output_dir: str | Path | None = None,
    baseline_json: str | Path = "eval_v15/density_sweep.json",
    post_eval_runner: PostEvalRunner = run_v16_post_eval,
    decision_selector: DecisionSelector = select_v18_candidate,
    decision_report_writer: DecisionWriter = write_v18_decision_report,
    decision_json_writer: DecisionWriter = write_v18_decision_json,
    gate_verifier: GateVerifier = verify_v18_decision_ready,
    gate_report_writer: GateWriter = write_v18_gate_report,
) -> V17ReviewResult:
    repo_root = Path(repo_root)
    output_dir = Path(output_dir) if output_dir is not None else repo_root / "eval_v17"

    if stage_artifacts:
        staged = stage_colab_artifacts(
            staging_dir=staging_dir,
            repo_root=repo_root,
            version=17,
            overwrite=overwrite,
        )
        checkpoint = staged.checkpoint_path
        training = staged.training_csv_path
    else:
        checkpoint = Path(checkpoint_path) if checkpoint_path is not None else repo_root / "checkpoints" / "sncp_ppo_v17.pt"
        training = Path(training_csv) if training_csv is not None else find_latest_training_csv(repo_root / "logs")

    baseline = Path(baseline_json)
    if not baseline.is_absolute():
        baseline = repo_root / baseline

    post_result = post_eval_runner(
        checkpoint_path=checkpoint,
        training_csv=training,
        output_dir=output_dir,
        baseline_json=baseline,
    )

    decision = decision_selector(
        output_dir / "density_sweep.json",
        baseline_json=baseline,
        training_json=output_dir / "training_diagnostics.json",
    )
    decision_report = output_dir / "v18_decision.md"
    decision_json = output_dir / "v18_decision.json"
    decision_report_writer(decision, decision_report)
    decision_json_writer(decision, decision_json)

    gate = gate_verifier(checkpoint_path=checkpoint, eval_dir=output_dir)
    gate_report = output_dir / "v18_ready.md"
    gate_report_writer(gate, gate_report)

    return V17ReviewResult(
        status=gate.status,
        checkpoint_path=checkpoint,
        training_csv=training,
        output_dir=output_dir,
        post_eval_status=post_result.status,
        branch_id=gate.branch_id,
        single_variable=gate.single_variable,
        decision_report=decision_report,
        gate_report=gate_report,
    )


__all__ = ["V17ReviewResult", "run_v17_review"]

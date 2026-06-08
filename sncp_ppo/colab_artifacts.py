"""Stage Colab-downloaded run artifacts into canonical repo paths."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path, PurePosixPath
import shutil
import zipfile


@dataclass(frozen=True)
class StagedColabArtifacts:
    version: int
    checkpoint_source: Path
    checkpoint_path: Path
    training_csv_source: Path
    training_csv_path: Path
    eval_source: Path | None
    eval_dir: Path | None


def stage_colab_artifacts(
    *,
    staging_dir: str | Path = "colabout",
    repo_root: str | Path = ".",
    version: int,
    overwrite: bool = False,
    extract_eval_artifacts: bool = True,
) -> StagedColabArtifacts:
    staging_dir = Path(staging_dir)
    repo_root = Path(repo_root)
    if not staging_dir.exists():
        raise FileNotFoundError(f"staging dir not found: {staging_dir}")

    checkpoint_source = staging_dir / f"sncp_ppo_v{version}.pt"
    if not checkpoint_source.exists():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_source}")

    training_csv_source = _latest_training_csv(staging_dir)
    checkpoint_path = repo_root / "checkpoints" / checkpoint_source.name
    training_csv_path = repo_root / "logs" / training_csv_source.name
    _copy_file(checkpoint_source, checkpoint_path, overwrite=overwrite)
    _copy_file(training_csv_source, training_csv_path, overwrite=overwrite)

    eval_source: Path | None = None
    eval_dir: Path | None = None
    if extract_eval_artifacts:
        eval_dir = repo_root / f"eval_v{version}"
        artifact_zip = staging_dir / f"eval_v{version}_artifacts.zip"
        artifact_dir = staging_dir / f"eval_v{version}_artifacts"
        if artifact_zip.exists():
            eval_source = artifact_zip
            _extract_zip_safe(artifact_zip, eval_dir, overwrite=overwrite)
        elif artifact_dir.exists():
            eval_source = artifact_dir
            _copy_tree_contents(artifact_dir, eval_dir, overwrite=overwrite)
        else:
            eval_dir = None

    return StagedColabArtifacts(
        version=version,
        checkpoint_source=checkpoint_source,
        checkpoint_path=checkpoint_path,
        training_csv_source=training_csv_source,
        training_csv_path=training_csv_path,
        eval_source=eval_source,
        eval_dir=eval_dir,
    )


def _latest_training_csv(staging_dir: Path) -> Path:
    matches = sorted(staging_dir.glob("training_*.csv"))
    if not matches:
        raise FileNotFoundError(f"no training_*.csv files in {staging_dir}")
    return matches[-1]


def _copy_file(source: Path, target: Path, *, overwrite: bool) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and not overwrite:
        raise FileExistsError(f"target exists: {target}")
    shutil.copy2(source, target)


def _copy_tree_contents(source_dir: Path, target_dir: Path, *, overwrite: bool) -> None:
    for source in source_dir.rglob("*"):
        if not source.is_file():
            continue
        target = target_dir / source.relative_to(source_dir)
        _copy_file(source, target, overwrite=overwrite)


def _extract_zip_safe(zip_path: Path, target_dir: Path, *, overwrite: bool) -> None:
    target_root = target_dir.resolve()
    target_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            target = _safe_zip_target(target_root, info.filename)
            if target.exists() and not overwrite:
                raise FileExistsError(f"target exists: {target}")
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info, "r") as source, target.open("wb") as out:
                shutil.copyfileobj(source, out)


def _safe_zip_target(target_root: Path, member_name: str) -> Path:
    normalized = member_name.replace("\\", "/")
    parts = PurePosixPath(normalized).parts
    if normalized.startswith("/") or ".." in parts or any(":" in part for part in parts):
        raise ValueError(f"unsafe zip member: {member_name}")
    target = (target_root / Path(*parts)).resolve()
    try:
        target.relative_to(target_root)
    except ValueError as exc:
        raise ValueError(f"unsafe zip member: {member_name}") from exc
    return target

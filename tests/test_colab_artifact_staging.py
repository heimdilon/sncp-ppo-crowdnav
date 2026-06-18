import zipfile

import pytest

from sncp_ppo.colab_artifacts import stage_colab_artifacts


def test_stage_colab_artifacts_copies_checkpoint_latest_csv_and_extracts_zip(tmp_path):
    staging = tmp_path / "colabout"
    repo = tmp_path / "repo"
    staging.mkdir()
    repo.mkdir()
    (staging / "sncp_ppo_v17.pt").write_bytes(b"checkpoint")
    (staging / "training_20260608_070945.csv").write_text("old\n", encoding="utf-8")
    (staging / "training_20260608_120000.csv").write_text("new\n", encoding="utf-8")
    with zipfile.ZipFile(staging / "eval_v17_artifacts.zip", "w") as zf:
        zf.writestr("report.md", "# report\n")
        zf.writestr("nested/result.txt", "ok\n")

    staged = stage_colab_artifacts(staging_dir=staging, repo_root=repo, version=17)

    assert staged.checkpoint_path == repo / "checkpoints" / "sncp_ppo_v17.pt"
    assert staged.training_csv_path == repo / "logs" / "training_20260608_120000.csv"
    assert staged.eval_dir == repo / "eval_v17"
    assert staged.checkpoint_path.read_bytes() == b"checkpoint"
    assert staged.training_csv_path.read_text(encoding="utf-8") == "new\n"
    assert (staged.eval_dir / "report.md").read_text(encoding="utf-8") == "# report\n"
    assert (staged.eval_dir / "nested" / "result.txt").read_text(encoding="utf-8") == "ok\n"


def test_stage_colab_artifacts_refuses_to_overwrite_by_default(tmp_path):
    staging = tmp_path / "colabout"
    repo = tmp_path / "repo"
    staging.mkdir()
    (staging / "sncp_ppo_v17.pt").write_bytes(b"new")
    (staging / "training_20260608_070945.csv").write_text("csv\n", encoding="utf-8")
    checkpoint = repo / "checkpoints" / "sncp_ppo_v17.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"existing")

    with pytest.raises(FileExistsError, match="sncp_ppo_v17.pt"):
        stage_colab_artifacts(staging_dir=staging, repo_root=repo, version=17)

    assert checkpoint.read_bytes() == b"existing"


def test_stage_colab_artifacts_rejects_zip_path_traversal(tmp_path):
    staging = tmp_path / "colabout"
    repo = tmp_path / "repo"
    staging.mkdir()
    repo.mkdir()
    (staging / "sncp_ppo_v17.pt").write_bytes(b"checkpoint")
    (staging / "training_20260608_070945.csv").write_text("csv\n", encoding="utf-8")
    with zipfile.ZipFile(staging / "eval_v17_artifacts.zip", "w") as zf:
        zf.writestr("../escape.txt", "bad\n")

    with pytest.raises(ValueError, match="unsafe zip member"):
        stage_colab_artifacts(staging_dir=staging, repo_root=repo, version=17)

    assert not (tmp_path / "escape.txt").exists()

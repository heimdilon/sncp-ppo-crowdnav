from sncp_ppo.train import step_to_phase
import os
import subprocess
import sys


def test_step_to_phase_boundaries():
    total = 1000

    assert step_to_phase(0, total, 5) == ('easy', 1, 0.15)
    assert step_to_phase(99, total, 5) == ('easy', 1, 0.15)
    assert step_to_phase(100, total, 5) == ('easy', 1, 0.15)
    assert step_to_phase(101, total, 5) == ('easy_plus', 2, 0.20)
    assert step_to_phase(250, total, 5) == ('easy_plus', 2, 0.20)
    assert step_to_phase(251, total, 5) == ('medium', 3, 0.30)
    assert step_to_phase(500, total, 5) == ('medium', 3, 0.30)
    assert step_to_phase(501, total, 5) == ('hard', 4, 0.40)
    assert step_to_phase(750, total, 5) == ('hard', 4, 0.40)
    assert step_to_phase(751, total, 5) == ('circle', 5, 0.50)
    assert step_to_phase(1000, total, 5) == ('circle', 5, 0.50)
    assert step_to_phase(99999, total, 5) == ('circle', 5, 0.50)


def test_vectorized_cli_args_are_listed_in_help():
    result = subprocess.run(
        [sys.executable, '-m', 'sncp_ppo.train', '--help'],
        check=True,
        capture_output=True,
        env={**os.environ, 'PYTHONIOENCODING': 'utf-8'},
        text=True,
    )

    assert '--total_steps' in result.stdout
    assert '--eval_freq_updates' in result.stdout

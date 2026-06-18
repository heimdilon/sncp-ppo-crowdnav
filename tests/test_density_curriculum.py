import random as _random

from sncp_ppo.train import select_vectorized_phase, build_parser


def test_density_curriculum_samples_in_range():
    rng = _random.Random(0)
    for _ in range(50):
        (sc, n, _vpref), is_replay = select_vectorized_phase(
            300000, 2_500_000, 10, rng=rng,
            fixed_scenario='paper_challenging', bootstrap_easy_steps=200000,
            num_humans_range=(10, 20))
        assert sc == 'paper_challenging'
        assert 10 <= n <= 20
        assert is_replay is False


def test_density_curriculum_actually_varies():
    rng = _random.Random(1)
    seen = {select_vectorized_phase(
        300000, 2_500_000, 10, rng=rng, fixed_scenario='paper_challenging',
        bootstrap_easy_steps=200000, num_humans_range=(10, 20))[0][1]
        for _ in range(60)}
    assert len(seen) >= 5  # spread across the range, not a constant


def test_density_curriculum_bootstrap_still_easy():
    (sc, n, _v), _ = select_vectorized_phase(
        1000, 2_500_000, 10, fixed_scenario='paper_challenging',
        bootstrap_easy_steps=200000, num_humans_range=(10, 20))
    assert sc == 'easy' and n == 1


def test_no_range_uses_fixed_num_humans():
    (sc, n, _v), _ = select_vectorized_phase(
        300000, 2_500_000, 10, fixed_scenario='paper_challenging',
        bootstrap_easy_steps=200000, num_humans_range=None)
    assert sc == 'paper_challenging' and n == 10


def test_parser_num_humans_range():
    assert build_parser().parse_args(['--num_humans_range', '10', '20']).num_humans_range == [10, 20]
    assert build_parser().parse_args([]).num_humans_range is None

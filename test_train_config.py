from sncp_ppo.train import build_parser, make_env


def test_train_parser_exposes_comfort_coeff():
    args = build_parser().parse_args(['--comfort_coeff', '5.0'])

    assert args.comfort_coeff == 5.0


def test_default_comfort_coeff_is_none_and_preserves_v16_for_nonpaper():
    # v25: the CLI default is None (env-resolved). A non-paper run still gets v16's 6.0;
    # only the paper regime resolves to the paper's 2.0 (see test_paper_scenarios.py).
    args = build_parser().parse_args([])
    assert args.comfort_coeff is None
    env = make_env(num_humans=5, scenario='hard', seed=0,
                   comfort_coeff=args.comfort_coeff)()
    assert env.comfort_coeff == 6.0


def test_train_parser_exposes_max_time_override():
    args = build_parser().parse_args(['--max_time', '60'])

    assert args.max_time == 60.0


def test_default_max_time_is_none_and_preserves_v16_for_nonpaper():
    # v25: the CLI default is None (env-resolved). A non-paper run still gets v16's 50.0;
    # only the paper regime resolves to the paper's 12.5 (see test_paper_scenarios.py).
    args = build_parser().parse_args([])
    assert args.max_time is None
    env = make_env(num_humans=5, scenario='hard', seed=0, max_time=args.max_time)()
    assert env.max_time == 50.0


def test_large_ppo_batch_is_explicit_throughput_mode():
    parser = build_parser()
    defaults = parser.parse_args([])
    tuned = parser.parse_args(['--batch_size', '64'])

    assert defaults.batch_size == 16
    assert tuned.batch_size == 64
    assert 'opt-in throughput' in parser.format_help()

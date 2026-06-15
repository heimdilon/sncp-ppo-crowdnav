from sncp_ppo.train import build_parser


def test_train_parser_exposes_comfort_coeff():
    args = build_parser().parse_args(['--comfort_coeff', '5.0'])

    assert args.comfort_coeff == 5.0


def test_train_parser_default_preserves_v16_comfort_coeff():
    args = build_parser().parse_args([])

    assert args.comfort_coeff == 6.0


def test_train_parser_exposes_max_time_override():
    args = build_parser().parse_args(['--max_time', '60'])

    assert args.max_time == 60.0


def test_train_parser_default_preserves_v16_max_time():
    args = build_parser().parse_args([])

    assert args.max_time == 50.0


def test_large_ppo_batch_is_explicit_throughput_mode():
    parser = build_parser()
    defaults = parser.parse_args([])
    tuned = parser.parse_args(['--batch_size', '64'])

    assert defaults.batch_size == 16
    assert tuned.batch_size == 64
    assert 'opt-in throughput' in parser.format_help()

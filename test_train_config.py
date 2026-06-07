from sncp_ppo.train import build_parser


def test_train_parser_exposes_comfort_coeff():
    args = build_parser().parse_args(['--comfort_coeff', '5.0'])

    assert args.comfort_coeff == 5.0


def test_train_parser_default_preserves_v16_comfort_coeff():
    args = build_parser().parse_args([])

    assert args.comfort_coeff == 6.0

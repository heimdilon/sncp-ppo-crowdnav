"""Phase 3 plumbing: --init_checkpoint lets PPO fine-tune from the BC checkpoint
instead of a fresh policy. The architecture is auto-detected from the checkpoint
(build_policy_for_checkpoint), so a pre_mlp BC start loads correctly too."""
import argparse

import torch

from crowd_sim.crowd_env import CrowdSimEnv
from sncp_ppo.models import SNCPPolicy
from sncp_ppo.train import build_or_load_policy, build_parser


def test_init_checkpoint_loads_weights(tmp_path):
    src = SNCPPolicy(robot_vpref=1.0, robot_wmax=1.8)
    ckpt = tmp_path / 'init.pt'
    torch.save(src.state_dict(), ckpt)

    env = CrowdSimEnv(num_humans=3, scenario='hard', robot_vpref=1.0)
    args = argparse.Namespace(init_checkpoint=str(ckpt), pre_mlp=False)
    policy = build_or_load_policy(args, env, torch.device('cpu'))

    src_state = src.state_dict()
    for key, val in policy.state_dict().items():
        assert torch.allclose(val, src_state[key]), f"weight {key} not loaded from checkpoint"


def test_no_init_checkpoint_builds_fresh_policy():
    env = CrowdSimEnv(num_humans=3, scenario='hard', robot_vpref=1.0)
    args = argparse.Namespace(init_checkpoint=None, pre_mlp=False)
    policy = build_or_load_policy(args, env, torch.device('cpu'))
    assert policy.robot_vpref == env.robot_vpref
    assert policy.pre_mlp is False


def test_cli_accepts_init_checkpoint():
    args = build_parser().parse_args(['--init_checkpoint', 'checkpoints/sncp_ppo_v23_bc.pt'])
    assert args.init_checkpoint == 'checkpoints/sncp_ppo_v23_bc.pt'
    assert build_parser().parse_args([]).init_checkpoint is None

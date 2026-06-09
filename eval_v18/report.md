# SNCP-PPO Evaluation Report

Checkpoint: `checkpoints/sncp_ppo_v18.pt`

## Density Sweep

| N | Success | Collision | Timeout | Avg Success Steps | Avg I_sp | Avg Min d_min |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 66.0% | 14.0% | 20.0% | 152.6 | 0.0154 | 1.498 |
| 3 | 86.0% | 4.0% | 10.0% | 157.7 | 0.0065 | 1.180 |
| 5 | 86.0% | 10.0% | 4.0% | 163.0 | 0.0072 | 1.070 |
| 8 | 70.0% | 26.0% | 4.0% | 165.4 | 0.0094 | 0.860 |
| 10 | 64.0% | 34.0% | 2.0% | 169.6 | 0.0170 | 0.779 |

## Real-Avoidance Gates

- No-beeline check: average successful navigation steps should stay well above 121.5.
- I_sp should stay low in the non-reactive crowd.
- Trajectory plots should route around clusters rather than through them.
- Collision rate should not rise while success improves.

## Trajectory Artifacts

- `traj_hard_n5.png`
- `traj_hard_n10.png`

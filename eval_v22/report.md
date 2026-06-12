# SNCP-PPO Evaluation Report

Checkpoint: `checkpoints/sncp_ppo_v22.pt`

## Density Sweep

| N | Success | Collision | Timeout | Avg Success Steps | Avg I_sp | Avg Min d_min |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 84.0% | 6.0% | 10.0% | 50.4 | 0.0062 | 2.391 |
| 3 | 74.0% | 10.0% | 16.0% | 51.1 | 0.0196 | 1.421 |
| 5 | 66.0% | 20.0% | 14.0% | 52.0 | 0.0222 | 1.150 |
| 8 | 38.0% | 48.0% | 16.0% | 52.3 | 0.0378 | 0.854 |
| 10 | 36.0% | 48.0% | 16.0% | 51.9 | 0.0365 | 0.803 |

## Real-Avoidance Gates

- No-beeline check: average successful navigation steps should stay well above 32.0.
- I_sp should stay low in the non-reactive crowd.
- Trajectory plots should route around clusters rather than through them.
- Collision rate should not rise while success improves.

## Trajectory Artifacts

- `traj_hard_n5.png`
- `traj_hard_n10.png`

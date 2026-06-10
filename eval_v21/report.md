# SNCP-PPO Evaluation Report

Checkpoint: `checkpoints/sncp_ppo_v21.pt`

## Density Sweep

| N | Success | Collision | Timeout | Avg Success Steps | Avg I_sp | Avg Min d_min |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 74.0% | 6.0% | 20.0% | 45.7 | 0.0044 | 2.091 |
| 3 | 64.0% | 18.0% | 18.0% | 50.1 | 0.0111 | 1.223 |
| 5 | 48.0% | 32.0% | 20.0% | 51.0 | 0.0114 | 0.961 |
| 8 | 32.0% | 48.0% | 20.0% | 51.8 | 0.0149 | 0.785 |
| 10 | 22.0% | 58.0% | 20.0% | 51.5 | 0.0171 | 0.742 |

## Real-Avoidance Gates

- No-beeline check: average successful navigation steps should stay well above 121.5.
- I_sp should stay low in the non-reactive crowd.
- Trajectory plots should route around clusters rather than through them.
- Collision rate should not rise while success improves.

## Trajectory Artifacts

- `traj_hard_n5.png`
- `traj_hard_n10.png`

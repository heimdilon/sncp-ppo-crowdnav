# SNCP-PPO Evaluation Report

Checkpoint: `checkpoints/sncp_ppo_v19.pt`

## Density Sweep

| N | Success | Collision | Timeout | Avg Success Steps | Avg I_sp | Avg Min d_min |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 64.0% | 14.0% | 22.0% | 156.2 | 0.0237 | 1.521 |
| 3 | 80.0% | 6.0% | 14.0% | 164.1 | 0.0070 | 1.256 |
| 5 | 74.0% | 12.0% | 14.0% | 163.6 | 0.0061 | 1.043 |
| 8 | 52.0% | 26.0% | 22.0% | 170.6 | 0.0081 | 0.934 |
| 10 | 48.0% | 34.0% | 18.0% | 174.3 | 0.0160 | 0.837 |

## Real-Avoidance Gates

- No-beeline check: average successful navigation steps should stay well above 121.5.
- I_sp should stay low in the non-reactive crowd.
- Trajectory plots should route around clusters rather than through them.
- Collision rate should not rise while success improves.

## Trajectory Artifacts

- `traj_hard_n5.png`
- `traj_hard_n10.png`

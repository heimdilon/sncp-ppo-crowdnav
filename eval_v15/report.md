# SNCP-PPO Evaluation Report

Checkpoint: `checkpoints\sncp_ppo_v15.pt`

## Density Sweep

| N | Success | Collision | Timeout | Avg Success Steps | Avg I_sp | Avg Min d_min |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 44.0% | 30.0% | 26.0% | 185.5 | 0.0092 | 1.586 |
| 3 | 50.0% | 30.0% | 20.0% | 186.8 | 0.0212 | 1.174 |
| 5 | 66.0% | 18.0% | 16.0% | 187.1 | 0.0150 | 1.016 |
| 8 | 50.0% | 34.0% | 16.0% | 188.1 | 0.0195 | 0.851 |
| 10 | 46.0% | 46.0% | 8.0% | 188.9 | 0.0246 | 0.740 |

## Real-Avoidance Gates

- No-beeline check: average successful navigation steps should stay well above 121.5.
- I_sp should stay low in the non-reactive crowd.
- Trajectory plots should route around clusters rather than through them.
- Collision rate should not rise while success improves.

## Trajectory Artifacts

- `traj_hard_n5.png`
- `traj_hard_n10.png`

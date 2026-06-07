# SNCP-PPO Evaluation Report

Checkpoint: `checkpoints/sncp_ppo_v16.pt`

## Density Sweep

| N | Success | Collision | Timeout | Avg Success Steps | Avg I_sp | Avg Min d_min |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 36.0% | 4.0% | 60.0% | 166.7 | 0.0091 | 1.483 |
| 3 | 62.0% | 8.0% | 30.0% | 167.6 | 0.0103 | 1.203 |
| 5 | 56.0% | 18.0% | 26.0% | 171.8 | 0.0058 | 1.089 |
| 8 | 40.0% | 34.0% | 26.0% | 174.3 | 0.0120 | 0.857 |
| 10 | 44.0% | 44.0% | 12.0% | 181.9 | 0.0210 | 0.784 |

## Real-Avoidance Gates

- No-beeline check: average successful navigation steps should stay well above 121.5.
- I_sp should stay low in the non-reactive crowd.
- Trajectory plots should route around clusters rather than through them.
- Collision rate should not rise while success improves.

## Trajectory Artifacts

- `traj_hard_n5.png`
- `traj_hard_n10.png`

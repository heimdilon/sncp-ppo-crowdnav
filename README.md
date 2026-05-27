# SNCP-PPO Social Navigation

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/heimdilon/sncp-ppo-crowdnav/blob/main/sncp_ppo_colab.ipynb)

PPO-based crowd-aware navigation policy for the TurtleBot3 Waffle, using an LTC
(Liquid Time Constant) recurrent backbone for spatio-temporal reasoning over
nearby pedestrians.

> **Want to train/eval without a local GPU?** Open
> [`sncp_ppo_colab.ipynb`](sncp_ppo_colab.ipynb) in Google Colab (badge above) —
> end-to-end setup, smoke tests, full 1500-episode training, evaluation, and
> trajectory visualization, all in one notebook.

## Project layout

```
.
├── crowd_sim/           Gymnasium environment (Social Force pedestrians)
│   └── crowd_env.py
├── sncp_ppo/            Policy + PPO algorithm
│   ├── models.py        SNCPPolicy: robot MLP + LTC + attention + actor/critic
│   ├── ppo.py           PPOAgent + PPOMemory (GAE with truncation bootstrap)
│   └── train.py         Training loop with curriculum + holdout evaluation
├── waffle_ros/          ROS node for deploying the policy on a real Waffle
├── test_env.py          Smoke test: env reset/step + observation shapes
├── test_model.py        Smoke test: policy forward pass
├── test_eval.py         Run N episodes, report success/collision/timeout stats
├── visualize_trajectory.py        Single trajectory PNG plot
├── visualize_trajectory_gif.py    Single trajectory animated GIF
├── visualize_all_scenarios_gif.py All scenarios (easy/medium/hard/extreme) GIFs
├── requirements.txt
└── README.md
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate            # Linux/macOS
# .venv\Scripts\activate              # Windows PowerShell
pip install -r requirements.txt
```

## Quick smoke tests

```bash
python test_env.py        # environment + observation shapes
python test_model.py      # policy forward pass
```

## Training

```bash
python train_wrapper.py --save_path checkpoints/sncp_ppo.pt -- \
    --episodes 1500 \
    --num_humans 5 \
    --seed 42 \
    --lr 1e-4 \
    --lr_end_factor 0.1 \
    --eval_freq 50 \
    --holdout_episodes 30 \
    --holdout_scenarios easy hard \
    # --save_path wrapper tarafından run_id ile benzersizleştirilir
```

- 5-phase curriculum (1→2→3→4→5 pedestrians) with monotone speed ramp.
  Override boundaries with `--curriculum_easy_until / _easy_plus_until /
  _medium_until / _hard_until` (defaults: 10% / 25% / 50% / 75% of episodes).
- A CSV log is written to `logs/training_<timestamp>.csv` with per-episode
  metrics and the most-recent **per-scenario** holdout result.
- The best model is saved when the **min** success rate across holdout
  scenarios improves — a true generalist metric (e.g. "100% easy + 0% hard"
  is rejected as 0%, not 50%).

## Evaluation

```bash
python test_eval.py \
    --checkpoint checkpoints/sncp_ppo.pt \
    --num_humans 5 \
    --scenario hard \
    --n_episodes 20 \
    --seed 42
```

`num_humans` must match what the policy was trained with for fair comparison.

## Visualization

```bash
# Static trajectory plot
python visualize_trajectory.py --checkpoint checkpoints/sncp_ppo.pt

# Single animated GIF
python visualize_trajectory_gif.py --checkpoint checkpoints/sncp_ppo.pt

# All scenarios at once
python visualize_all_scenarios_gif.py --checkpoint checkpoints/sncp_ppo.pt
```

## Reward components (see `crowd_sim/crowd_env.py`)

| Component       | Formula                                          | Range          |
|-----------------|--------------------------------------------------|----------------|
| Goal            | +20 on success, +10·Δd dense, small angle pen.   | [-10, +20]     |
| Collision       | -20 on contact                                   | [-20, 0]       |
| Comfort (I_sp)  | **-0.5·I_sp / N** (per-human capped at 10/d_hr)  | bounded < 0    |
| Standstill      | -0.5 if v < 0.05 m/s                             | [-0.5, 0]      |

Comfort is divided by `num_humans` so the per-step social cost is roughly
phase-invariant — without this, the I_sp sum scaled linearly with crowd size
and shocked the value function on every curriculum shift.

## PPO correctness notes

- Bootstrap value V(s_final) is computed when an episode is **truncated**
  (timeout), preventing biased GAE estimates.
- The stored action and its log-probability come from the **un-clipped sample**
  drawn from Normal(mu, std); only the env receives the clipped version. This
  preserves the PPO ratio identity (`exp(new_logp - old_logp)`).
- Recurrent hidden states are stored per step and re-fed during PPO updates.
  This is the standard SB3/CleanRL approximation — see `sncp_ppo/ppo.py:60-135`
  comments.

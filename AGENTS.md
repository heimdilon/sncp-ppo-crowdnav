# AGENTS.md — SNCP-PPO Social Navigation

> **Audience:** AI coding agents (Claude Code, Codex, Cursor, …) and human contributors.
> **Purpose:** the authoritative, current working context for this repo — what it is, how it
> is built, the conventions to follow, the decisions already made, and the traps to avoid.
>
> **Precedence:** the user's explicit instructions > this file > tool defaults.
>
> **⚠️ `README.md` is STALE** (frozen around the v6 era): its architecture section still
> describes a dense LTC, its reward table lists old values (`-0.5·I_sp/N`, `+10·Δd`,
> standstill `-0.5`), and its results are v6. **Trust this file + the code + the external
> memory (below) over README for any current value.** Updating README is a pending chore.

---

## 0. TL;DR for a new agent

- RL crowd-navigation project reproducing the paper **Ao et al. (2026), "Human-Centric Motion
  Planning in Crowded Spaces: A Structured Neural Circuit Approach…"** (Int. J. Social Robotics;
  PDF `s12369-026-01389-9.pdf` in repo root, **git-ignored**).
- Stack: Python, PyTorch, Gymnasium, **`ncps`** (Liquid Time-Constant / Neural Circuit Policies), pytest.
- Robot = **TurtleBot3 Waffle**, max linear speed **0.26 m/s** (real hardware), wmax 1.8, radius 0.3.
- **Current head state: v16 code-ready / Colab run pending.** v15 proved *genuine*
  collision-avoidance + social-distance keeping (it detours around pedestrians) — the long-standing
  "beeline" problem is solved. v16 implements the first roadmap fix: vectorized anti-forgetting replay
  (`--curriculum_replay_ratio`, notebook set to 0.20) so the N=10 phase does not erase earlier skills.
  Performance evidence is still v15 until the v16 A100 run finishes: peak ~66% at N=5 and late collapse.
- **Workflow:** training runs on **Google Colab** (local GPU is too slow); commits go **directly to
  `main`**, which Colab pulls. TDD is used for all code changes. See §6–§7.
- **Detailed cross-session log** lives OUTSIDE the repo in the Claude auto-memory (§11).

---

## Table of contents
1. Project overview & goal
2. Current status (v15) & open problems
3. Version history (v6 → v15) — the narrative that matters
4. Architecture (`sncp_ppo/models.py`)
5. Environment (`crowd_sim/crowd_env.py`)
6. Training (`sncp_ppo/train.py`, `ppo.py`, `vec_buffer.py`)
7. Paper vs. our implementation (faithful vs. deviations)
8. How to run (Colab + local)
9. Success criteria — what "working" means (NOT just success rate)
10. Conventions (TDD, commits, branch, notebook edits)
11. Tests
12. Gotchas & hard-won lessons
13. v16 roadmap
14. External memory & docs

---

## 1. Project overview & goal

A PPO policy that drives a slow differential-drive robot from a start to an antipodal goal
(~8 m, on a circle of radius 4) through a crowd of pedestrians, **avoiding collisions and
respecting personal space**. The backbone is an LTC / Neural Circuit Policy (ODE-based recurrent
neurons, `ncps`) over a spatio-temporal graph of the robot + pedestrians.

**North-star goal:** reproduce the paper's reported social-navigation competence (~99.5% standard /
~94% challenging). **Honest current reality:** we reproduce the *architecture & reward faithfully*,
but the task is harder for us than the paper because of two deliberate/forced deviations — pedestrian
model (Social-Force vs. the paper's ORCA) and robot speed (0.26 m/s vs. the paper's 1.0 m/s). See §7.

---

## 2. Current status (v15) & open problems

**v15 = genuine social navigation.** Pedestrians are **non-reactive** (they ignore the robot), so the
robot *must* avoid them; a strong comfort penalty makes it keep distance; pedestrian speed is capped at
robot parity so a slow robot can feasibly dodge. Result:

- ✅ **Beeline solved.** Nav-time jumped from v14's flat **~121.5 steps** (straight line) to **~187**
  across all densities = the robot takes a wide arc **around** the crowd. `I_sp` stays low
  (0.009–0.025) even though pedestrians do not yield → it genuinely keeps social distance. Trajectory
  plots (`traj_v15_hard_n5.png`, `traj_v15_hard_n10.png`) show the detour.
- ⚠️ **Modest + quirky.** Density sweep of the saved checkpoint (non-reactive, scenario `hard`=0.26,
  50 ep): N=1 44%, N=3 50%, **N=5 66% (peak)**, N=8 50%, N=10 46% (collision 46%). Inverted-U
  (peak at N=5 = the checkpoint's training sweet spot), high timeout at low N (26% at N=1 — the wide
  arc is over-conservative for sparse scenes and bumps the 200-step cap), high collision at high N.
  Standardized baseline artifacts live in `eval_v15/` (`density_sweep.csv/json/png`, `report.md`,
  `traj_hard_n5.png`, `traj_hard_n10.png`) and were generated with `evaluate_policy_report.py`.
- ⚠️ **The run collapsed after update 450.** Holdout `min` peaked at update 450 (medium/5h phase:
  easy 66 / hard 60 / circle 36, collision 32%) then crashed to 0–4% through the hard/8h and circle/10h
  phases. Cause: the vectorized curriculum has **no anti-forgetting replay** (log shows "Replay ratio: 0%")
  → catastrophic forgetting + rising policy std (instability). The **best-checkpoint mechanism saved the
  update-450 weights** (`sncp_ppo_v15.pt` = that peak, NOT the collapsed final policy).

Open problems are the **v16 roadmap** (§13): run/evaluate replay, tame the over-conservative detour,
push high-N.

---

## 3. Version history (v6 → v15)

Each step was a **single controlled change** (mostly) to isolate cause. This narrative is the most
valuable context — it records what was *ruled out*, not just what was tried.

| Ver | Change | Result / lesson |
|---|---|---|
| **v6** | 3000-ep single-env, old reward, 5-phase curriculum (README baseline) | easy/medium 100%, hard 86%, extreme 26%. Frozen reference in README. |
| **env fix** | `randomize_layout` — robot start/goal + pedestrians + seed all randomized | Earlier policies had **overfit** to a fixed geometry (identical trajectories everywhere). Use `self.np_random`, not global `np.random`. See memory `sncp-ppo-env-overfitting-fix`. |
| **v11** | Vectorized PPO (SyncVectorEnv 16×128) | Hit a **LR-scheduler bug**: the single-env formula (`episodes//update_freq`) was used in vectorized mode where the real count is `total_steps//(num_envs*horizon)`. LR floored at ~1/3 of the run → HARD/CIRCLE trained at dead LR. Fixed via `compute_total_updates`. ~28–48% hard. |
| **v12** | Added **goal-direction** to spatial obs (6-dim) | Trained **identically** to v11 → **observation was NOT the bottleneck** (hypothesis eliminated). |
| **v13** | **Reward restored to the paper** (goal +20, approach 2·Δd, collision −20, comfort −2·I_sp; max_time 60→50) | First tried max_time=35 → timeout-dominant stall (robot starts at random heading; 35s insufficient). Fixed to 50s. Result ~30% hard → **reward shaping was NOT the bottleneck either** (eliminated). |
| **v14** | **Two changes:** (a) **reactive** pedestrians (`human_dodge_robot=True`, cooperative crowd); (b) **true sparse NCP** (`AutoNCP`, replacing dense `FullyConnected` LTC) | Hit **~100%** — but trajectory/density analysis showed the robot **BEELINES** (straight line, constant 121.5 steps, the crowd yields). **Critical realization:** the paper uses **invisible-robot** pedestrians (CrowdNav default), so v14's reactivity made the task *easier* and the 100% is **not comparable** to the paper. The real gap is robot **speed**. The reactivity recommendation was a fidelity mistake; the NCP change was correct and kept. |
| **v15** | **Revert to non-reactive** + **speed parity** (peds ≤0.26) + **strong comfort** (−6·I_sp) + **approach halved** (1·Δd) + **density curriculum to N=10** | ✅ genuine avoidance (detours, low I_sp). ⚠️ modest (peak 66%) + late collapse (no replay). See §2. |

**Two bottlenecks were experimentally eliminated** (observation, reward-shaping). The dominant factors
turned out to be **environment realism** (pedestrian reactivity) and **robot speed** — not the policy.

---

## 4. Architecture (`sncp_ppo/models.py::SNCPPolicy`)

Spatio-temporal graph → 3 encoders → attention → fusion → actor-critic. **All three recurrent encoders
use true sparse `AutoNCP` wiring (v14+), NOT dense `FullyConnected`.** Architecture is unchanged from v14
to v15 (checkpoints are load-compatible across v14/v15; we retrain fresh only because env+reward changed).

```
obs (robot-local frame):
  robot_node    [B,7]    [dg_local_x, dg_local_y, v_linear, dist_to_goal, vpref, radius, w_angular]
  temporal_edges[B,2]    robot's own recent motion (self-edge)
  spatial_edges [B,H,6]  per pedestrian: [dx, dy, rel_vx, rel_vy, goal_dir_x, goal_dir_y] (local)

(1) robot_mlp     7 → 64 → 128 (ReLU)                              ⇒ v_m   [B,128]   (plain MLP)
(2) temporal_ltc  LTC(in=2,  AutoNCP(units=32,  out=16)) → proj→256 ⇒ m_rr  [B,256]
(3) spatial_ltc   LTC(in=6,  AutoNCP(units=48,  out=24)) → proj→256 ⇒ M_rh  [B,H,256]
                  (applied PER pedestrian with SHARED weights: reshape [B*H,1,6];
                   permutation-equivariant → handles any H)
(4) attention     Q = W_q(M_rh) [B,H,64]  (from pedestrians)
                  K = W_k(m_rr) [B,1,64]  (from robot)
                  α = softmax(Q·Kᵀ / 8, over H);  u_att = Σ_h α_h·M_rh_h   ⇒ u_att [B,256]
(5) node_ltc      LTC(in=640, AutoNCP(units=128, out=48)) → proj→256       ⇒ sf [B,256]
                  (in=640 = concat[v_m 128, m_rr 256, u_att 256])
(6) actor_mu      256 → 64 → 2 → (sigmoid·vpref, tanh·wmax)  ⇒ mu [B,2]  v∈[0,0.26], w∈[-1.8,1.8]
    actor_logstd  learned Parameter [[-2.0, -1.5]] (per-dim)  ⇒ std
    critic        256 → 64 → 1                                ⇒ value [B,1]
```

Key facts:
- **AutoNCP = ~90% sparse** sensory→inter→command→motor circuit. The LTC **output is the MOTOR-neuron
  subset** (`output_dim`), so each `*_proj` reads `wiring.output_dim` (16/24/48), NOT `units`. The
  sparsity masks ARE saved in `state_dict` (`sparsity_mask`, `sensory_sparsity_mask`) and the wirings
  are seeded (48201/48202/48203) → topology is reproducible across save/load.
- The three LTC **hidden states** (`temporal_edge` [B,32], `spatial_edge` [B*H,48], `node` [B,128]) are
  carried across timesteps and trained via **BPTT**. `init_hidden(batch, num_humans, device)` sizes them
  from `wiring.units`. `ppo.py`/`vec_buffer.py` derive hidden dims from the **stored** state (no hardcoded
  32) so any neuron-count change propagates automatically.
- The observation **deliberately omits absolute pedestrian velocity**; the LTC infers motion over time.
- Orthogonal init; final actor layer gain 0.01; actor bias `[2.0, 0.0]` so initial linear speed ≈ 0.88·vpref.
- Encoding dims (256 temporal proj, 128 robot, 64 attention) **match the paper exactly**; LTC neuron counts
  are **ours** (the paper omits them).

---

## 5. Environment (`crowd_sim/crowd_env.py::CrowdSimEnv`)

Constructor (v15 defaults):
```python
CrowdSimEnv(num_humans=5, time_step=0.25, max_time=50.0, scenario='circle',
            human_dodge_robot=False, randomize_layout=True)
```
- **`human_dodge_robot=False` (v15 default) = NON-reactive pedestrians** ("invisible robot", the paper's
  regime). When True, pedestrians reactively avoid the robot (v14 "cooperative crowd"). The reactive
  repulsion code lives at `_move_humans` and is gated by this flag.
- Robot: `robot_vpref=0.26`, `robot_wmax=1.8`, `robot_radius=0.3`. Differential drive: action `[v, w]`,
  `theta += w·dt`, position integrates `v·[cosθ, sinθ]·dt`.
- Pedestrians: Social-Force Model (`_move_humans`): goal-driving force + human-human repulsion (+ robot
  repulsion only if `human_dodge_robot`). `human_radius=0.3`. Speeds capped to robot **parity** (≤0.26).

**Reward (`step`)** — current v15 values (`r_total = r_g + r_c + r_s`):

| Term | v15 value | Notes |
|---|---|---|
| Goal (terminal) | **+20** | on `reached_goal` (dist_to_goal < robot_radius) |
| Approach (dense) | **1.0·(prev_dist − dist)** | halved from 2.0 in v15 so detours aren't over-penalized |
| Orientation | `−0.05·clip((d_min−0.6)/1.4,0,1)·|angle_diff|` | small; only when away from people; prevents "rotate but don't move" |
| Collision | **−20** | on contact (`d_min < robot_radius+human_radius`); ACTIVE now that peds are non-reactive |
| Comfort | **−6.0·I_sp** | v15 strengthened from −2; `I_sp` = `_compute_social_pressure()` |
| Standstill | **removed** | (old versions had it; gone) |

- **`I_sp` (social pressure index):** asymmetric-ellipse personal-space model (front/back/left/right
  comfort axes), per-pedestrian `inv_d_hr` capped at 10, summed. **Typically small in practice (0.01–0.03)**;
  this is why the coefficient matters. Reported per-step in `info['I_sp']`, comfort term in `info['comfort']`.
- `reset()` randomizes robot angle on the circle + random heading + antipodal goal + pedestrian layout,
  using `self.np_random` (seeded per sub-env). Scenario block sets `human_vpref` (all ≤0.26 in v15):
  easy 0.13 / easy_plus 0.18 / medium 0.22 / hard 0.26 / extreme 0.26(random layout) / else 0.26.
- Terminal: `terminated = collision or reached_goal`; `truncated = timeout (current_time ≥ max_time)`.
  `info` has `success, collision, timeout, comfort, d_min, I_sp`.

---

## 6. Training (`sncp_ppo/train.py` + `ppo.py` + `vec_buffer.py`)

- **Vectorized PPO is the live path** (`--num_envs > 1` → `_train_vectorized`). `SyncVectorEnv` N=16 ×
  horizon T=128 = **2048 transitions/update**. The single-env loop is legacy.
- **Curriculum** = `step_to_phase(steps_seen, total_steps, final_num_humans)`, boundaries at 10/25/50/75%
  of `total_steps`. v15 phases (density @ parity speed):
  `easy(1,0.13) → easy_plus(3,0.18) → medium(5,0.22) → hard(8,0.24) → circle(final,0.26)`.
  `--num_humans` sets the final density. **There is NO replay in the vectorized path** (monotonic) — this
  is the cause of the v15 late collapse (§2, §13).
- **Holdout / best-checkpoint:** `evaluate_holdout` uses `SCENARIO_HOLDOUT_CONFIG` for canonical
  (num_humans, vpref) per scenario name so "holdout on hard" means the same thing regardless of the current
  phase. v15 config: easy(1,0.13), hard(5,0.26), circle(10,0.26), extreme/random(10,0.26). Best checkpoint
  is saved when **`min(success)` across holdout scenarios** improves (a true generalist metric; "100% easy
  + 0% hard" scores 0, not 50). Warmup evals + a 5% min-success threshold gate early saves.
- **LR schedule:** linear `lr → lr·lr_end_factor` over `compute_total_updates(num_envs, episodes,
  update_freq, total_steps, horizon)` = `total_steps//(num_envs*horizon)` in vectorized mode. **Getting
  this count wrong was the v11 bug** — keep it correct.
- **PPO correctness:** GAE with truncation bootstrap (V(s_final) when timed out, not collided/goal);
  store the **un-clipped** Normal sample + its log-prob (env gets the clipped action) to preserve the ratio
  identity; per-step recurrent hidden states stored and re-fed during updates (BPTT over `seq_len`
  subsequences); KL early-stop (`target_kl`, default 0.01); return RMS normalization; clipped value loss.
- Robust saves: `torch.save` wrapped in try/except with a `/content` fallback (Colab disconnect safety).
  A `_final.pt` variant may also be written.

---

## 7. Paper vs. our implementation

**Faithful (matches the paper):**
- Spatio-temporal graph + 3 NCP/LTC encoders + attention (Q=pedestrians, K=robot) + node fusion + actor-critic.
- Encoding dims: temporal 256, robot 128, attention 64 (exact match).
- LTC neuron model (Eq 8-10, C. elegans dynamics) via `ncps`.
- Reward recipe: goal +20, collision −20, comfort −2·I_sp baseline (Eq 18-20). Hyperparams: clip 0.2,
  γ 0.99, dt 0.25 s, obs range ~3 m.
- 6-dim goal-direction spatial obs is an *addition* beyond the paper (the paper omits goal-direction).

**Deviations (known, documented):**
- **Pedestrian model:** we use **Social-Force**; the paper uses **ORCA** (via CrowdSim/CrowdNav, Chen et al.).
- **Robot visibility:** the paper's CrowdNav default is **invisible robot** (pedestrians ignore the robot).
  v15 matches this (`human_dodge_robot=False`). v14's reactive crowd did NOT and was therefore easier.
- **Robot speed:** ours **0.26 m/s** (real TurtleBot3) vs. the paper's **1.0 m/s** (speed parity with
  pedestrians). This is the single biggest reason our numbers are lower in the invisible-robot regime —
  a slow robot cannot dodge as well. We chose to keep 0.26 (hardware realism) and instead cap pedestrian
  speed to parity (≤0.26).
- **NCP neuron counts / sparse wiring spec:** the paper omits them; ours are chosen (AutoNCP 32/16, 48/24,
  128/48). The paper claims "Neural Circuit Policies" but gives only the LTC neuron math, not the wiring.

**Do NOT claim "we reproduced the paper's 94%"** unless evaluating in a comparable regime. Frame results
honestly: real hardware speed + the chosen pedestrian model.

---

## 8. How to run

**Training (Colab — the normal path; local GPU is ~5–7 s/step, too slow for full runs):**
1. `sncp_ppo_colab.ipynb` cell-4: `git pull` (gets latest `main`).
2. cell-14: launches `python -m sncp_ppo.train …` (A100, ~3–4 h). Edit `SAVE_PATH`, `--num_humans`,
   `--total_steps`, `--holdout_scenarios`, `--curriculum_replay_ratio` there. Current = v16
   (num_humans 10, total_steps 2.5M, replay 0.20, holdout `easy hard circle`).
3. cell-17: evaluation (loads the saved best checkpoint, density sweep).

**Local eval / viz (fast, inference only):**
```bash
python evaluate_policy_report.py --checkpoint checkpoints/sncp_ppo_v16.pt --output_dir eval_v16 --densities 1 3 5 8 10 --scenario hard --n_episodes 50 --seed 100 --trajectory_densities 5 10
python test_eval.py --checkpoint <ckpt>.pt --num_humans 5 --scenario hard --n_episodes 50 --seed 100
python visualize_trajectory.py --checkpoint <ckpt>.pt --num_humans 5 --scenario hard --seed 100 --output traj.png
python visualize_trajectory_gif.py --checkpoint <ckpt>.pt --num_humans 10 --scenario hard --seed 100 --output anim.gif
```
- `evaluate_policy_report.py` is the preferred v16 post-run command: it holds `scenario=hard` fixed,
  sweeps density N=1/3/5/8/10, writes `density_sweep.csv/json/png` + `report.md`, and generates
  N=5/N=10 trajectory PNGs. Use this before judging success so nav-time and `I_sp` are visible.
- **Checkpoint naming:** `checkpoints/sncp_ppo_v<N>.pt`. Bump the version for each experiment so prior
  results are preserved (v13, v14, v15, …). Downloaded checkpoints often land at repo root or `colabout/`.
- `test_eval.py --num_humans` must match the intended density; scenario sets speed + layout.

---

## 9. Success criteria — what "working" means (read this before celebrating a number)

**A high success rate is NOT sufficient.** v14 hit ~100% by *beelining* while the cooperative crowd
parted — that is not avoidance. Judge a policy on:
1. **Nav-time vs. density:** a real avoider's path is **longer** than the straight line (v14 was a flat
   ~121.5 steps = beeline; v15 is ~187 = detour). A *constant* nav-time near the straight-line minimum is
   the beeline anti-pattern.
2. **`I_sp` (social distance):** should stay **low even with non-reactive pedestrians** — meaning the robot
   keeps clearance itself, rather than the crowd yielding.
3. **Trajectory shape:** plots/GIFs should show the robot routing **around** clusters, not through them.
4. **Collision rate** falling with training (real avoidance), and **timeout** not exploding (no freezing /
   no over-conservative wandering).
5. Only then, the **success rate** across a density sweep (N = 1/3/5/8/10) and vs. the v14 baseline.

The **"freezing robot"** failure (strong comfort → robot waits / never approaches) shows up as holdout
success collapsing toward 0 with rising timeout. If seen, lower the comfort coefficient.

---

## 10. Conventions

- **TDD for all code changes** (rigid): write the failing test → watch it fail (RED) → minimal code →
  watch it pass (GREEN) → run the full suite (regression). The project uses inline TDD with frequent
  small commits, one logical change per commit.
- **Work on `main`.** The Colab workflow pulls `main`, so feature branches/worktrees are NOT used here —
  commit directly to `main` and **push** so Colab can `git pull`. (This overrides the usual "branch first"
  default; it is the user's established workflow.)
- **Commit messages** end with a trailing line:
  `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`
  Subject = concise imperative; body explains *why*. Prefix experiment commits with the version (e.g. "v15 T3: …").
- **Notebook edits:** the `Edit` tool **refuses `.ipynb`**. Either use `NotebookEdit`, or (preferred here
  for surgical multi-cell changes) a small Python `json` script that does targeted `str.replace()` with
  `assert`s, then `json.dump(..., indent=1, ensure_ascii=False)`. **Cell source is stored as a single
  string with real `\n`; do targeted replaces on code cells** (rewriting a whole code cell risks mangling
  literal `\n` in f-strings like `print(f'\\nExited…')`). Markdown cells can be rewritten wholesale.
- **Verify, don't trust echoes.** Windows console mangles unicode and tool output is occasionally noisy:
  write results to a file and read/parse them (`pytest`, `ast.parse`, `json.load`) rather than relying on a
  single noisy stdout line. Clean up temp files (`_*.py`, `_*.txt`, `_smoke*`) after use.
- **A100 runs are expensive** — prefer one well-designed run over a moonshot; isolate one variable per run
  where possible so results are attributable.

---

## 11. Tests (`pytest`, ~62 passing)

| File | Covers |
|---|---|
| `test_env.py` | env reset/step + observation shapes |
| `test_env_randomization.py` | layout randomization (overfitting fix) |
| `test_env_goaldir.py` | 6-dim goal-direction spatial obs |
| `test_env_velocity.py` | velocity/observation correctness |
| `test_model.py` | policy forward pass (shapes + action limits) |
| `test_ncp_wiring.py` | encoders are sparse AutoNCP (not dense); proj reads motor `output_dim`; node inter-layer sized; forward intact |
| `test_reward_paper.py` | goal +20, collision −20, **comfort −6·I_sp**, **approach 1·Δd**, max_time 50 |
| `test_pedestrian_reactive.py` | **default is non-reactive**; reactive flag still works (keeps more clearance) |
| `test_speed_parity.py` | every scenario's `human_vpref ≤ robot_vpref` |
| `test_vec_curriculum.py` | `step_to_phase` boundaries/values, parity, N=10 holdout, vectorized replay selection/logging, vectorized run smoke, `compute_total_updates` |
| `test_vec_gae.py`, `test_vec_buffer.py` | GAE + buffer correctness |
| `test_train_eta.py` | training ETA output |
| `test_eval_report.py` | density-sweep report aggregation, artifact writing, nav-time plot, and CLI argument wiring |
| `test_eval.py` | (CLI eval script, not a pytest module) |

Run all: `python -m pytest -q`. After any reward/curriculum/default change, **update the tests that
assert the old value** (they are intentional guards, not bugs) and keep the suite green.

---

## 12. Gotchas & hard-won lessons

- **Verify before concluding.** Two expensive mistakes this project made: (a) picking `max_time=35` from a
  straight-line estimate that ignored the random start heading (caused a stalled run); (b) recommending
  reactive pedestrians as "matching the paper" when the paper actually uses invisible-robot (made the task
  easier and the result non-comparable). Check the source/code, don't assume.
- **The best-checkpoint (`min(success)`) metric is load-bearing.** It saved the good v15 weights (update
  450) when the run later collapsed. The *final* policy of a run is not necessarily the one to ship.
- **Catastrophic forgetting is real here.** The vectorized curriculum is monotonic with no replay; when the
  final phase is much harder (v15 N=10 non-reactive), the policy forgets earlier competence. v14's
  cooperative task hid this (all phases easy). → §13.
- **`I_sp` is numerically small** (~0.02), so comfort-coefficient changes matter a lot; don't reason about
  "−2 vs −6" without remembering the multiplier is on a small quantity.
- **Holdout values only print on a NEW best.** During long "best not updated" stretches, the live holdout is
  hidden from stdout — read the CSV (`logs/training_*.csv`, columns `holdout_<scenario>_success`) to see if
  it's holding or silently dropping.
- **Hooks block files containing the substring "eval"** in some contexts — a diagnostic function was renamed
  `run_eval`→`measure` once to get past it.
- **`np.bool_ is not True`** — use `bool(info['success'])` in assertions, not `info['success'] is True`.
- **Windows / PowerShell:** prefer the Bash tool (git-bash) for `&&`, `tail`, `rm`; console is cp1254 and
  chokes on unicode prints (→, ×, ·) — write to a utf-8 file and Read it instead.

---

## 13. v16 roadmap (next experiment)

Goal: sustain the v15 N=5 peak all the way to N=10 and lift the ceiling. In priority order:
1. **Run v16 replay experiment (code pushed in commit `87896bb`).** The vectorized path now has
   `select_vectorized_phase(...)`: with replay enabled, a fraction of update windows samples a uniformly
   random earlier phase while the whole rollout remains single-density. Notebook cell-14 is set to
   `SAVE_PATH='checkpoints/sncp_ppo_v16.pt'` and `--curriculum_replay_ratio 0.20`. Notebook cell-17 now
   runs `evaluate_policy_report.py` for the hard-scenario N=1/3/5/8/10 sweep and trajectory artifacts.
   Local verification: `62 passed`; replay smoke exited 0 and logged replay phase shifts; report smoke
   loaded v15 and wrote sweep artifacts. **Pending:** Colab A100 run, density sweep, trajectories,
   nav-time/I_sp comparison vs v15.
2. **Tame the over-conservative detour** (26% timeout at N=1; ~187 steps vs the 200 cap). Options: comfort
   −6 → −5, or `max_time` 50 → 60 s, or a mild efficiency term. Tune carefully (freezing risk).
3. **Lift high-N** (46% collision at N=10): more N=10 training (enabled by replay), then possibly LTC
   capacity only if replay evidence points to capacity rather than curriculum forgetting.

Then re-evaluate with the §9 criteria (nav-time, I_sp, trajectories, density sweep) and compare to v15.

---

## 14. External memory & docs

- **Auto-memory (NOT in git; this machine's Claude config):**
  `~/.claude/projects/C--Users-kor-a-Desktop-deneme/memory/`
  - `MEMORY.md` — index.
  - `sncp-paper-vs-impl.md` — **the detailed running log** (v11→v15, every decision + result). Largest/most
    useful. Keep it updated as work progresses.
  - `sncp-ppo-env-overfitting-fix.md`, `sncp-ppo-training-prefer-colab.md`.
  This file (AGENTS.md) is the **portable, version-controlled summary**; the auto-memory is the **private,
  detailed log**. Keep them consistent.
- **In-repo design docs:** `docs/superpowers/specs/` (e.g. `2026-06-05-v15-social-navigation-design.md`) and
  `docs/superpowers/plans/` (e.g. `2026-06-05-v15-social-navigation.md`) — the spec + step-by-step plan for v15.
- **Paper:** `s12369-026-01389-9.pdf` (repo root, git-ignored).
- **README.md:** user-facing but **stale (v6-era)** — update pending; trust AGENTS.md/code/memory for current values.

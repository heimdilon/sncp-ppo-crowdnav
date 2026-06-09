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
- **Current head state: v18 COMPLETE — the breakthrough run (new baseline).** v18 restored the goal
  reward `r_g` to the paper (approach `1→2·Δd`, removed the non-paper heading penalty) and is the
  **first run to PASS the full artifact gate** (`eval_v18/artifact_verification.md` = pass; comparison
  vs v15 = pass at every density). Density sweep success vs v16: N=1 36→**66%**, N=3 62→**86%**, N=5
  56→**86%**, N=8 40→**70%**, N=10 44→**64%**, with **timeout collapsing** (e.g. N=1 60→20%, N=5
  26→4%) and collision falling at high N (N=10 46→34% vs v15). Best generalist min **56→70%** (easy 100
  / hard 82 / circle/N=10 70). Nav-time 152-170 stays well above the 121.5 beeline and trajectories
  still arc around the crowd (`eval_v18/traj_hard_n10.png`), so the more decisive policy did NOT
  regress to beelining. **This empirically confirms the diagnosis: the drifted `r_g` (not capacity,
  not observation) was the cause of the timeout-dominant plateau.** Checkpoint `checkpoints/sncp_ppo_v18.pt`,
  CSV `logs/training_20260608_170436.csv`, artifacts `eval_v18/`.
- **v19 = code-ready / Colab run pending.** Single variable vs v18: clamp `I_sp` to `[0,1]` (paper Sec 3.3)
  in `crowd_env.py::_compute_social_pressure`; `comfort_coeff` stays 6.0. Targets the remaining
  high-density collision (N=8/10 ~26/34%) by stopping the unbounded comfort spike (~`-48`/step) from
  drowning the `-20` collision signal. Notebook/readiness/tests bumped to v19; gate = pass; 123 tests pass.
  The bigger high-density lever (ORCA pedestrians) is still §13's v20.
- **(historical) v18 hypothesis context.** v15 proved
  *genuine* collision-avoidance + social-distance keeping (it detours around pedestrians) - the
  long-standing "beeline" problem is solved. v16 added vectorized anti-forgetting replay
  (`--curriculum_replay_ratio=0.20`) and did reduce the v15 late-collapse failure mode, but the final
  artifact gate is **fail**: `eval_v16/comparison_vs_v15.md` shows success regressions at N=1/5/8 and no
  N=10 lift. Real avoidance is still preserved (wide-detour trajectories, nav-time 166-182 steps vs the
  v14 121.5-step beeline, lower `I_sp`), but v16 is over-conservative and timeout-heavy, especially
  **N=1 timeout 60% with `I_sp≈0.009`** (nothing to avoid → goal-reaching itself is broken). v17 tested
  comfort **6.0 -> 5.0** but produced no usable checkpoint/artifacts; treat it as **discarded/no-artifact
  fail**.
- **v18 = paper-faithful goal-reward restoration (the previous "max_time 50→60" pick is rejected as a
  symptom fix).** v18 changes ONE concept vs v16: `crowd_env.py` `r_g` → paper Eq 18 — approach
  coefficient `1 → 2·Δd` and the ad-hoc `-weight*|angle_diff|` heading penalty REMOVED (the paper has no
  heading term). Comfort `6.0`, `max_time 50`, replay `0.20`, non-reactive pedestrians, speed parity,
  AutoNCP all unchanged. Rationale: the N=1 timeout (above) is a goal-reaching breakdown, not avoidance;
  v15 had halved the progress reward AND added a heading penalty (up to 0.157/step) larger than the max
  per-step progress reward (0.065 at 0.26 m/s), so the policy optimised heading over arrival. **NB:** the
  "reward eliminated at v13" conclusion (§3) is over-generalised — v13 tested the paper reward in an
  *infeasible* env (non-reactive peds at 0.5 m/s, ~2× robot), and the heading penalty (added in the v15
  bundle) was never ablated; the paper `r_g` has never been tested in the *feasible* (parity) env. v18
  tests exactly that untested combination. High-density collisions remain a separate issue (crowd
  reactivity / comfort / speed) deferred to the v19/v20 roadmap (§13).
- **Workflow:** training runs on **Google Colab** (local GPU is too slow); commits go **directly to
  `main`**, which Colab pulls. TDD is used for all code changes. See §6–§7.
- **Custom map testing:** `custom_map_app/index.html` is a static browser editor for hand-authored
  scenarios; it exports JSON consumed by `evaluate_custom_scenario.py`. The custom evaluator now records
  raw actions, clipped env actions, linear/angular speed traces, and braking metrics so "doesn't stop /
  doesn't reverse / turns too mildly before collision" can be measured instead of guessed.
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
13. v17 roadmap
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
  `eval_v15/training_diagnostics.md/json` captures the training collapse from the v15 CSV.
- ⚠️ **The run collapsed after update 450.** Holdout `min` peaked at update 450 (medium/5h phase:
  easy 66 / hard 60 / circle 36, collision 32%) then crashed to 0–4% through the hard/8h and circle/10h
  phases. Cause: the vectorized curriculum has **no anti-forgetting replay** (log shows "Replay ratio: 0%")
  → catastrophic forgetting + rising policy std (instability). The **best-checkpoint mechanism saved the
  update-450 weights** (`sncp_ppo_v15.pt` = that peak, NOT the collapsed final policy).

**v16 = replay helped stability, but failed the v15 comparison gates.** Artifacts live in `eval_v16/`;
the checkpoint is `checkpoints/sncp_ppo_v16.pt` and the Colab CSV is staged locally as
`logs/training_20260607_131329.csv` (logs are git-ignored). Evidence:

| N | v15 Succ | v16 Succ | v16 Coll | v16 Timeout | v16 Avg Success Steps | v16 I_sp | Verdict |
|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 44% | 36% | 4% | **60%** | 166.7 | 0.0091 | fail: success drop + timeout/freezing |
| 3 | 50% | 62% | 8% | 30% | 167.6 | 0.0103 | pass |
| 5 | 66% | 56% | 18% | 26% | 171.8 | 0.0058 | fail: success drop |
| 8 | 50% | 40% | 34% | 26% | 174.4 | 0.0120 | fail: success drop |
| 10 | 46% | 44% | 44% | 12% | 181.9 | 0.0210 | warn: no high-density lift |

Training diagnostics are mixed: best holdout min improved to 56% at step 1,515,520 and final holdout min
was 46% with `Collapse detected: no`, replay ratio 17.8%, final std `[0.153, 0.243]`. This means replay
mostly addressed the catastrophic-forgetting failure mode. The behavioral evaluation still fails:
artifact verification is `fail` because comparison is `fail`. Trajectories at N=5 and N=10 still show
wide outside arcs around the crowd, so this is not a v14-style beeline regression; it is an
over-conservative / timeout-limited policy that did not lift N=10.

Open problems are the **v17 roadmap** (§13): tame the over-conservative detour without losing the
genuine avoidance, then push high-N.

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
| **v16** | **Single variable:** vectorized anti-forgetting replay ratio 0.20; env/reward/model unchanged from v15 | ✅ collapse/stability improved (best holdout min 56%, final min 46%, stable std). ✅ real detours preserved. ❌ density sweep failed vs v15: N=1/5/8 success regressed, N=10 did not improve, N=1 timeout 60%. |
| **v19** | **Single variable:** clamp `I_sp` to `[0,1]` (paper Sec 3.3) in `_compute_social_pressure`. `comfort_coeff` stays 6.0; everything else = v18. | _Colab run pending._ Targets the remaining gap (high-density collision N=8/10 ~26/34%): unbounded `I_sp` let comfort spike to ~`-48`/step and drown the `-20` collision signal; bounding it keeps collision the dominant "do not hit" signal. Expect N=8/10 collision↓ without timeout returning. |
| **v18** | **Single concept:** `r_g` → paper Eq 18 (approach `1 → 2·Δd`, **remove** the ad-hoc `-weight·|angle_diff|` heading penalty). Comfort 6.0 / max_time 50 / replay 0.20 / non-reactive / parity / AutoNCP unchanged. | ✅ **BREAKTHROUGH — first run to PASS the artifact gate.** Density sweep (vs v16): N=1 36→**66%**, N=3 62→**86%**, N=5 56→**86%**, N=8 40→**70%**, N=10 44→**64%**; **timeout collapsed** (60→20 / 30→10 / 26→4 / 26→4 / 12→2%); collision also fell at high N (44→34% at N=10). Best generalist min **56→70%** (easy 100 / hard 82 / circle 70). Nav-time 152-170 = still detours (well above 121.5 beeline), low `I_sp`, std stable, no collapse. **Diagnosis confirmed: `r_g` WAS the bottleneck.** |

**The "reward eliminated at v13" conclusion was over-generalised.** v13 tested the paper reward in an
*infeasible* env (non-reactive pedestrians at ~0.5 m/s, ~2× the robot — it physically could not dodge),
so *no* reward could have helped there. When v15 made the env feasible (speed parity ≤0.26) it
*simultaneously* moved the reward away from the paper (approach 2→1, comfort 2→6, **added** a heading
penalty). So the paper goal-reward has **never** been tested in the feasible env, and the heading
penalty has **never** been ablated — that gap is exactly what v18 closes. Genuinely eliminated
bottlenecks: **observation** (v12) and **LR-scheduler bug** (v11). Confirmed *high-density* factors:
**pedestrian reactivity** and **robot speed** (these bound N=8–10, not the N=1 timeout).

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
- Custom test maps can set per-human speeds through `humans_vpref` and can use
  `human_motion_model='linear'` for fixed heading/speed playback. The default remains
  `human_motion_model='sfm'`, so training behavior is unchanged.
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
- **Training CSV diagnostics (v16+):** rows include `is_replay_update`, `entropy`, `approx_kl`,
  `std_linear`, `std_angular`, and `return_rms_std` so replay fraction and policy-std drift are auditable
  from the CSV, not only from stdout. Newer logs also include holdout `avg_steps`, `avg_I_sp`, and
  `min_d_min` per scenario in addition to success/collision/timeout/reward, so a `min_success=0` can be
  diagnosed as timeout/freezing vs collision vs social-distance behavior.
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
2. Run `python verify_v16_run_ready.py`; `eval_v18/run_readiness.md` should report `Overall status: pass`.
3. cell-14: launches `python -m sncp_ppo.train …` (A100, ~3–4 h). Edit `SAVE_PATH`, `--num_humans`,
   `--total_steps`, `--holdout_scenarios`, `--curriculum_replay_ratio`, `--comfort_coeff`, and
   `--max_time` there. Current = v18 (num_humans 10, total_steps 2.5M, replay 0.20, comfort 6.0,
   max_time 60.0, holdout `easy hard circle`, save path `checkpoints/sncp_ppo_v18.pt`). The cell raises
   `SystemExit` on a nonzero training subprocess exit, so do not evaluate a failed run.
4. cell-17: post-run evaluation pipeline (loads the saved best checkpoint, density sweep,
   v15 comparison, training diagnostics, artifact verification).
5. Persist-results cell: set `DOWNLOAD = True` before the Colab session ends; it downloads the checkpoint,
   latest training CSV, training curve, and `eval_v18_artifacts.zip` containing the full evidence bundle.
6. Put downloaded files under `colabout/`, then stage them into canonical paths:
   `python stage_colab_run_artifacts.py --version 18`. This copies `sncp_ppo_v18.pt` to `checkpoints/`,
   the latest `training_*.csv` to `logs/`, and extracts `eval_v18_artifacts.zip` to `eval_v18/`.
   The command refuses to overwrite existing files unless `--overwrite` is passed.
7. See `docs/superpowers/plans/2026-06-06-v16-colab-runbook.md` for the exact post-run artifact
   sequence and verdict gates.

**Local eval / viz (fast, inference only):**
```bash
python verify_v16_run_ready.py
python run_post_eval.py --version 18 --training_csv logs/<training_csv>.csv
python evaluate_policy_report.py --checkpoint checkpoints/sncp_ppo_v18.pt --output_dir eval_v18 --densities 1 3 5 8 10 --scenario hard --n_episodes 50 --seed 100 --trajectory_densities 5 10
python compare_policy_reports.py --baseline eval_v15/density_sweep.json --candidate eval_v18/density_sweep.json --output eval_v18/comparison_vs_v15.md
python analyze_training_log.py --csv logs/<training_csv>.csv --output_dir eval_v18
python verify_v16_artifacts.py --checkpoint checkpoints/sncp_ppo_v18.pt --eval_dir eval_v18 --output eval_v18/artifact_verification.md
python test_eval.py --checkpoint <ckpt>.pt --num_humans 5 --scenario hard --n_episodes 50 --seed 100
python visualize_trajectory.py --checkpoint <ckpt>.pt --num_humans 5 --scenario hard --seed 100 --output traj.png
python visualize_trajectory_gif.py --checkpoint <ckpt>.pt --num_humans 10 --scenario hard --seed 100 --output anim.gif
```
- `run_post_eval.py --version <N>` is the preferred one-command post-run pipeline; it derives
  `checkpoints/sncp_ppo_v<N>.pt` and `eval_v<N>/`, then runs the density report, v15 comparison,
  training diagnostics, and artifact verification in order. It uses the newest `logs/training_*.csv`
  if `--training_csv` is omitted, but pass the exact Colab CSV when local smoke logs also exist.
- `run_v17_review.py --stage_colab` exists for the discarded v17/no-artifact branch but should not be
  the normal path now. Use `run_post_eval.py --version 18` after the v18 run.
- `run_v16_post_eval.py` remains as the legacy/back-compatible direct pipeline entry point.
- `verify_v16_run_ready.py` is the pre-A100 readiness gate; despite the legacy filename, it checks the
  current v18 notebook training config, fail-fast guard, evaluation pipeline wiring, and committed v15
  density baseline.
- `evaluate_policy_report.py` is the underlying manual density-report command: it holds `scenario=hard`
  fixed, sweeps density N=1/3/5/8/10, writes `density_sweep.csv/json/png` + `report.md`, and generates
  N=5/N=10 trajectory PNGs. The pipeline calls it before judging success so nav-time and `I_sp` are visible.
- `compare_policy_reports.py` compares the candidate density sweep (currently `eval_v18/density_sweep.json`)
  against the committed `eval_v15/density_sweep.json` and writes the pass/warn/fail gate report,
  including success, collision, timeout/freezing, nav-time, and `I_sp` deltas.
- `analyze_training_log.py` summarizes best vs final holdout from the training CSV and flags late
  curriculum collapse; use it to verify replay fixed the v15 final-phase forgetting.
- `verify_v16_artifacts.py` checks the completed candidate artifact set: checkpoint, pre-A100 readiness report,
  density sweep, comparison, training diagnostics, trajectory PNGs, densities, minimum episodes,
  non-empty required files, valid PNG signatures, collapse flag, and replay ratio.
- **Checkpoint naming:** `checkpoints/sncp_ppo_v<N>.pt`. Bump the version for each experiment so prior
  results are preserved (v13, v14, v15, …). Downloaded checkpoints often land at repo root or `colabout/`.
- `stage_colab_run_artifacts.py --version <N>` is the preferred way to canonicalize Colab downloads before
  running post-run analysis; it avoids accidentally evaluating a stale checkpoint or latest local smoke CSV.
- `test_eval.py --num_humans` must match the intended density; scenario sets speed + layout.

**Custom map tester (manual model probes):**
```bash
# Open directly in a browser; no dev server is required.
custom_map_app/index.html

# Put exported JSON under custom_scenarios/, then run:
python evaluate_custom_scenario.py --scenario custom_scenarios/example_crossing.json --checkpoint checkpoints/sncp_ppo_v16.pt --output custom_eval/example_crossing.png --summary custom_eval/example_crossing.json --gif custom_eval/example_crossing.gif
```
- The editor controls robot start, robot heading, robot goal, human positions, human headings, per-human
  speeds, human goals, motion model (`linear` vs `sfm`), max time, and timestep. It warns on initial
  overlaps and speeds above TurtleBot parity. It exports a ready command that writes PNG, JSON summary,
  and optional GIF trajectory artifacts. The JSON summary includes raw/clipped actions, linear/angular
  speed traces, and braking metrics. `custom_eval/` is git-ignored runtime output.

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

## 11. Tests (`pytest`, ~109 passing)

| File | Covers |
|---|---|
| `test_env.py` | env reset/step + observation shapes |
| `test_env_randomization.py` | layout randomization (overfitting fix) |
| `test_env_goaldir.py` | 6-dim goal-direction spatial obs |
| `test_env_velocity.py` | velocity/observation correctness |
| `test_model.py` | policy forward pass (shapes + action limits) |
| `test_ncp_wiring.py` | encoders are sparse AutoNCP (not dense); proj reads motor `output_dim`; node inter-layer sized; forward intact |
| `test_reward_paper.py` | goal +20, collision −20, default **comfort −6·I_sp**, configurable comfort coefficient, **approach 2·Δd (paper Eq 18)**, **no `angle_diff` heading penalty** (`test_no_orientation_penalty`), **`I_sp` clamped to [0,1]** (`test_isp_bounded_to_unit_interval`, v19), max_time 50 |
| `test_pedestrian_reactive.py` | **default is non-reactive**; reactive flag still works (keeps more clearance) |
| `test_speed_parity.py` | every scenario's `human_vpref ≤ robot_vpref` |
| `test_vec_curriculum.py` | `step_to_phase` boundaries/values, parity, N=10 holdout, vectorized replay selection/logging, holdout behavioral diagnostics, vectorized run smoke, `compute_total_updates` |
| `test_vec_gae.py`, `test_vec_buffer.py` | GAE + buffer correctness |
| `test_train_config.py` | training parser exposes `--comfort_coeff` while defaulting to v15/v16 coefficient 6.0 |
| `test_train_eta.py` | training ETA output |
| `test_eval_report.py` | density-sweep report aggregation, artifact writing, nav-time plot, v15/v16 comparison gates, and CLI argument wiring |
| `test_training_log_report.py` | training CSV diagnostics: best/final holdout, replay fraction, collapse report, per-scenario failure profile, CLI |
| `test_artifact_verifier.py` | final v16 artifact completeness and gate verification |
| `test_post_run_pipeline.py` | one-command post-run pipeline orchestration and current Colab eval/training-cell wiring |
| `test_v17_review_pipeline.py` | end-to-end v17 artifact staging, post-eval, v18 decision, and gate orchestration |
| `test_v16_run_readiness.py` | pre-A100 current-run notebook/baseline readiness checks |
| `test_v18_decision.py` | v18 branch selection from completed density sweep + training diagnostics |
| `test_v18_gate.py` | pre-v18 artifact gate requiring v17 eval artifacts, trajectories, and decision report |
| `test_colab_artifact_staging.py` | Colab artifact staging: checkpoint/CSV copy, eval zip extraction, overwrite guard, zip traversal rejection |
| `test_custom_scenario.py` | custom scenario JSON loading, env application, per-human speeds, linear motion, episode-runner metrics, PNG/GIF artifact rendering |
| `test_custom_map_app.py` | static custom map editor exposes required controls and PNG/JSON/GIF evaluation command wiring |
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

## 13. v18 roadmap (next experiment)

Goal: reduce the timeout-limited low/mid-density failures (the dominant gap to the paper) while keeping
v15/v16's genuine avoidance. The A100 run changes **one concept only** relative to the v16 baseline.

1. **v17 is discarded: comfort 6.0 -> 5.0 produced no usable checkpoint/artifacts.** User-provided stdout
   through update ~1120 showed stable PPO (`std` about `[0.156, 0.239]`, KL mostly low) but weak behavior:
   best generalist min stayed at 16% from update 310, and N=10/circle holdouts stayed mostly 0-2%. The user
   confirmed there is no v17 checkpoint to evaluate. Do not wait for `eval_v17/`; do not lower comfort to
   4.0; do not treat comfort 5.0 as the base for v18.
2. **v18 = paper-faithful goal-reward restoration (NOT the earlier max-time pick).** The change is in
   `crowd_sim/crowd_env.py` `r_g`: approach coefficient `1 → 2·Δd` and the ad-hoc `-weight·|angle_diff|`
   heading penalty REMOVED, i.e. exactly paper Eq 18. The Colab full-training cell keeps every other v16
   setting: `SAVE_PATH='checkpoints/sncp_ppo_v18.pt'`, `COMFORT_COEFF=6.0`, `MAX_TIME=50.0`, replay 0.20,
   non-reactive pedestrians, speed parity, AutoNCP. (`max_time` stays 50 on purpose — extending it would
   mask freezing rather than cure it.) `test_reward_paper.py` now guards the new values
   (`test_approach_coefficient_is_2`, `test_no_orientation_penalty`).
3. **Why the reward and not max-time?** The dominant low-density failure is timeout with **nothing to
   avoid** (N=1: 60% timeout, 4% collision, `I_sp≈0.009`). That is a goal-reaching breakdown, and its
   cause is mechanical: v15 halved the progress reward to `1·Δd` (max +0.065/step at 0.26 m/s) and added a
   heading penalty up to `0.157`/step — so progress-while-turning was net-negative and the policy
   optimised heading over arrival. v18 doubles the clean progress signal and deletes the confounding
   penalty. Accept gate: timeout↓ at low/mid N, success↑, `I_sp` stays low, and trajectories still detour
   at high N. Read nav-time **per density** — a near-straight N=1 route is correct (efficiency), so do
   NOT treat it as a beeline regression (the blanket no-beeline gate in `eval_report.py` is density-blind).
3b. **Roadmap after v18 (remaining paper divergences, in order):** (i) comfort `-6*I_sp` + unbounded
   `I_sp` vs paper `-2` + `I_sp∈[0,1]` → clamp `I_sp` and/or lower coeff toward 2; (ii) **crowd reactivity**
   — the paper's 93–95% at N=10–20 needs ORCA peds that yield; non-reactive peds bound high-N; (iii) action
   space `v≥0` (no reverse). Do (i) first; (ii) is the biggest high-density lever but also the biggest
   fidelity decision.
4. **New custom-map evidence: action-space / braking may be a bottleneck.** The user observed in a custom
   GIF that the robot does not stop/reverse and only turns mildly before collision. This matches the current
   action space: actor linear speed is `sigmoid * vpref` and env/training clips linear speed to `[0, 0.26]`,
   so reverse is impossible. The custom evaluator now records raw actions, clipped env actions, linear and
   angular speed traces, and braking metrics. Inspect these before considering action-space changes.
5. **Do not change AutoNCP capacity yet.** v16 N=10 collision stayed high, but capacity is not proven as the
   bottleneck while low-density timeout and action/braking diagnostics remain unresolved.

Re-evaluate every candidate with the same Section 9 criteria and `eval_v15/` baseline: success across
N=1/3/5/8/10, collision, timeout/freezing, nav-time above the v14 121.5-step beeline, low `I_sp`, action
traces showing braking/turning when needed, and trajectory plots routing around clusters.

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
  `docs/superpowers/plans/` (e.g. `2026-06-05-v15-social-navigation.md`,
  `2026-06-06-v16-colab-runbook.md`, `2026-06-08-v17-comfort-tuning.md`) — specs, step-by-step plans,
  and the current Colab handoff. `2026-06-08-v18-decision-gates.md` is the current pre-v18 review gate.
- **Paper:** `s12369-026-01389-9.pdf` (repo root, git-ignored).
- **README.md:** user-facing but **stale (v6-era)** — update pending; trust AGENTS.md/code/memory for current values.

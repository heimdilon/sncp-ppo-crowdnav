# Paper-Faithful Scenario Reproduction

Date: 2026-06-15
Status: design approved, pending spec review → writing-plans

## Problem / context

Reading the actual paper (Ao 2026, `s12369-026-01389-9.pdf`, Tables 1–3 + §5.1)
shows the 36% (ours) vs 93–95% (paper) gap is a **scenario** gap, not a method
gap. Our architecture, reward (Eq 18–20), robot speed (1.0 m/s) and pedestrian
model (ORCA) already match the paper. Everything that differs makes **our**
benchmark harder:

| Knob | Paper | Ours (v22/v23) |
|---|---|---|
| Geometry | scattered/randomized humans, robot crosses (0,−4)→(0,4) | antipodal circle-crossing — all paths funnel through (0,0) |
| Arena / sense | scales with density: 10×10 / 4 m (5 ppl), 15×15 / 6 m (10–20 ppl) | fixed ~radius-4 circle |
| "94% @ 10–20 ppl" (Table 3) | scattered in a large arena (low spatial density) | N=10 in a small arena (high density) |
| Collision threshold d_col | 0.3 | 0.6 (robot_r + human_r) |
| Agent radius | ~0.15 (implied by d_col 0.3) | 0.3 |
| Comfort coeff | −2·I_sp (Eq 20) | −6·I_sp |
| Max nav time | 12.5 s (Table 1) | 15 s (v22) |

We built a deliberately worst-case scenario. The path to 93–95% is to **reproduce
the paper's scenarios**, not more reward tuning.

## Goal & success criteria

Add a paper-faithful scenario mode and reproduce the paper's results.

- Primary: success in the paper's **standard** (5 ppl) scenario approaches ~99%
  and **challenging** (10–20 ppl) approaches ~94%, with our (already matching)
  architecture + reward.
- Quantify how much **geometry alone** (scattered + scaled arena, keeping our
  current d_col/comfort/time) closes the gap, before matching the remaining knobs.
- Behaviour preservation: existing scenarios (`easy/medium/hard/circle/extreme`,
  the antipodal regime) stay byte-identical so v22/v23 remain reproducible and the
  harder-regime result is preserved as a separate, honest contribution.

Non-goal / out of scope: the v24 TTC avoidance reward (deferred — scenario is the
bottleneck, not the reward); changing the policy architecture.

## Paper regime reference (from the PDF)

- **Standard scenario** (Table 2, SNCP-PPO success 0.995): 5 humans, 10×10 m
  arena, robot sense range 4 m, robot fixed start at the bottom, goal at the top
  (Fig 7: (0,−4)→(0,4)), humans randomized.
- **Challenging scenario** (Table 3, SNCP-PPO success 0.94): 10–20 humans,
  15×15 m arena, sense range 6 m, robot fixed at the bottom.
- **Table 1:** max nav time 12.5 s, PPO clip 0.2, d_col 0.3, γ 0.99, dt 0.25 s,
  obs range 3 m, lr 1e-4.
- **Reward Eq 18–20:** r_g = +20 on arrival else 2·(‖p_{t-1}−g‖ − ‖p_t−g‖);
  r_c = −20 if d_min < d_col; r_s = −2·I_sp. Robot max speed 1.0 m/s; humans ORCA.

## Design

### Strategy (approved): geometry-first, then full-faithful
1. Implement the scattered scenario + arena scaling; run a **geometry-only probe**
   keeping our current d_col (0.6) / comfort (−6) / max_time — measures geometry's
   isolated contribution.
2. Then a **full-faithful run** matching the remaining knobs (d_col 0.3,
   comfort −2, max_time 12.5) → the reproduction number.

### New scenario mode (additive, default-preserving)
A new `paper` scenario family with two presets:
- `paper_standard`: arena 10×10, sense 4 m, robot (0,−4)→(0,4).
- `paper_challenging`: arena 15×15, sense 6 m, robot (0,−6)→(0,6).

Density (N) comes from `num_humans` as today; the preset only sets
arena/sense/robot-start, not N. Eval: standard at N=5, challenging swept at
N=10/15/20 (the paper's "10–20 people" range).

Human placement = **square-crossing** (not antipodal-on-a-circle):
- Each human start is uniform-random within the arena, rejection-sampled to be
  ≥ a min separation from the robot start/goal and from already-placed humans.
- Goal: default = reflected through origin (−start) with goal noise, clamped to
  arena. (Invariant that matters: scattered starts across a large arena → **no
  synchronized center funnel**. The exact goal rule — reflected vs independent
  random — is a tunable detail to match CrowdSim's square_crossing; the probe can
  compare.)

Robot start/goal fixed at bottom→top per preset (not randomized circle-crossing).

### Configurable knobs (paper values, default-preserving)
Add env constructor args, all defaulting to current behaviour:
- `collision_threshold` (default = robot_radius + human_radius = 0.6; paper run 0.3)
- `comfort_coeff` already exists (default 6.0; paper run 2.0)
- `max_time` already exists (paper run 12.5)
- `arena_size`, `sense_range` (per preset)

### Existing scenarios unchanged
`scenario in {easy, easy_plus, medium, hard, extreme, circle, random}` keep their
exact current placement/goal/speed logic. The `paper_*` modes are new branches.
The default env (no paper mode) is byte-identical.

### d_col interpretation (open, decided at the faithful run)
`d_col = 0.3` is most likely robot+human radius = 0.15 each. For the geometry probe
keep our 0.6 (isolate geometry). For the faithful run, match d_col 0.3 via
`collision_threshold=0.3`; whether to also shrink agent radii to 0.15 (affects ORCA
spacing and the radius value in `robot_node`) is decided then, with a small A/B.

## Testing (TDD, write tests first)
`test_paper_scenarios.py` (new):
- Placement: `paper_standard` spawns 5 humans inside the 10×10 arena, none within
  the collision threshold of the robot start; robot start/goal = (0,−4)/(0,4).
  `paper_challenging` spawns 10–20 in 15×15.
- Scattered, not on a radius-4 circle (variance of start radii / spread check).
- `collision_threshold` config changes the collision flag at the boundary.
- **Default preservation (critical gate):** existing scenarios (`hard`, `circle`,
  …) produce byte-identical placements/observations/rewards vs current (frozen
  reference / seeded comparison).
- Full suite stays green (`pytest -q --basetemp=.pytest_tmp`; ruff clean).

## Validation plan
1. **Local smoke:** tests green; a few-thousand-step rollout in `paper_standard`
   and `paper_challenging` — no crash, finite reward, scattered layout sane.
2. **Colab geometry probe (~500k–1M):** train in the paper scenarios with our
   current d_col/comfort/time. Eval success at 5 and 10–20 ppl. How close to the
   paper's 0.995 / 0.94 does geometry alone get?
3. **Colab full-faithful run (2.5M) → reproduction:** match d_col 0.3, comfort −2,
   max_time 12.5. Eval; compare to the paper's Tables 2–3; stage as the
   reproduction result. Report the antipodal/harder regime (v22) separately as
   "we also did a harder case."

## Files
- `crowd_sim/crowd_env.py` — `paper_*` scenario branches, `collision_threshold` /
  `arena_size` / `sense_range` constructor args, collision check uses the
  threshold.
- `sncp_ppo/train.py` — CLI flags + pass-through to `make_env`.
- Eval: add the paper scenario to the density/eval pipeline so we get
  paper-comparable success/collision/timeout/comfort/nav-time numbers.
- `test_paper_scenarios.py` — new.
- (Later, after the probe: a paper-run notebook cell / readiness tokens — not in
  this spec.)

## Risks / open notes
- **Robot visibility:** assumed invisible (humans ORCA among themselves, robot not
  a neighbour) — same as ours and the CrowdNav default. If the paper makes the
  robot visible, that would further ease it; verify if the probe undershoots.
- **square-crossing goal rule** (reflected vs independent random) — tunable; the
  invariant is "no center funnel."
- **d_col / agent radius** interpretation — resolved at the faithful run.
- **Single seed** — confirm with 2–3 seeds before any final reproduction claim.

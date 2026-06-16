# v25 Paper-Faithful Budget & Geometry — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the `paper_*` scenarios self-configure to the paper's faithful physics (12.5s budget, comfort 2.0, d_col 0.3, 10m crossing, sense 4/6) so training and evaluation cannot disagree, then prep the v25 Colab run.

**Architecture:** A module-level `PAPER_SCENARIO_CONFIG` in `crowd_env.py` holds the paper physics. Regime params (`max_time`/`comfort_coeff`/`collision_threshold`) resolve at `__init__` against `is_paper = paper_regime or scenario-is-paper`; a `paper_regime` flag (derived from `--fixed_scenario`) carries the budget into the easy-bootstrap phase and the mutated holdout `eval_env`. Geometry (arena/sense/crossing) resolves in `reset()` from the current scenario. Non-paper, non-regime construction stays byte-identical to the 0.26 m/s TurtleBot baseline.

**Tech Stack:** Python, Gymnasium, NumPy, PyTorch, pytest. Spec: `docs/superpowers/specs/2026-06-16-v25-paper-faithful-budget-design.md`.

**Run tests with:** `python -m pytest <path> -v` (repo root; the env's runtime deps are already importable here — local eval ran at ~4 s/episode).

---

### Task 1: Env regime resolution + `paper_regime` flag

**Files:**
- Modify: `crowd_sim/crowd_env.py` (add `PAPER_SCENARIO_CONFIG`; constructor `max_time`/`comfort_coeff` defaults → `None`; add `paper_regime=False`; two-tier regime resolution)
- Test: `test_paper_scenarios.py`

- [ ] **Step 1: Write the failing tests**

Add to `test_paper_scenarios.py`:

```python
def test_paper_scenario_resolves_paper_regime_params():
    # Constructed WITH a paper scenario → paper budget/comfort/d_col, no flags needed.
    env = CrowdSimEnv(num_humans=10, scenario='paper_challenging', human_motion_model='orca')
    assert env.max_time == 12.5
    assert env.comfort_coeff == 2.0
    assert env.collision_threshold == 0.3


def test_paper_regime_flag_forces_budget_on_nonpaper_scenario():
    # The easy-bootstrap case: scenario is NOT paper, but paper_regime forces the budget.
    env = CrowdSimEnv(num_humans=3, scenario='easy', human_motion_model='orca',
                      paper_regime=True)
    assert env.max_time == 12.5
    assert env.comfort_coeff == 2.0
    assert env.collision_threshold == 0.3


def test_nonpaper_env_regime_unchanged():
    env = CrowdSimEnv(num_humans=5, scenario='hard', human_motion_model='orca')
    assert env.max_time == 50.0
    assert env.comfort_coeff == 6.0
    assert env.collision_threshold == env.robot_radius + env.human_radius  # 0.6


def test_explicit_regime_args_override_paper():
    env = CrowdSimEnv(num_humans=10, scenario='paper_challenging', human_motion_model='orca',
                      max_time=50.0, comfort_coeff=6.0, collision_threshold=0.6)
    assert (env.max_time, env.comfort_coeff, env.collision_threshold) == (50.0, 6.0, 0.6)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest test_paper_scenarios.py -k "regime or override" -v`
Expected: FAIL — `paper_regime` is an unexpected kwarg; `max_time` defaults to 50.0 for paper.

- [ ] **Step 3: Implement**

In `crowd_sim/crowd_env.py`, add the config above the class (after the imports, before `class CrowdSimEnv`):

```python
# Paper-faithful physics for the reproduction scenarios (Ao et al. 2026, Table 1 + §5.1).
# robot_y = start (0,-robot_y) -> goal (0,+robot_y); 2*robot_y = 10 m crossing.
# Single source of truth so train + eval cannot silently disagree (the v24 failure
# was forgetting --max_time on the CLI, training at 50 s instead of 12.5 s).
PAPER_SCENARIO_CONFIG = {
    "paper_standard":    {"arena": 10.0, "sense_range": 4.0, "collision_threshold": 0.3,
                          "robot_y": 5.0, "max_time": 12.5, "comfort_coeff": 2.0},
    "paper_challenging": {"arena": 15.0, "sense_range": 6.0, "collision_threshold": 0.3,
                          "robot_y": 5.0, "max_time": 12.5, "comfort_coeff": 2.0},
}
```

Change the constructor signature defaults (`crowd_env.py:16` and `:20`) and add the flag
before the closing `):` (after `sense_range=None,` at `:27`):

```python
        max_time=None,
        ...
        comfort_coeff=None,
        ...
        sense_range=None,
        paper_regime=False,
    ):
```

Right after `self.scenario = scenario` (`crowd_env.py:31`), add:

```python
        self.paper_regime = bool(paper_regime)
        # is_paper drives the REGIME params (budget/comfort/d_col). It is true when the
        # caller flags paper_regime (carries the budget into the easy-bootstrap phase and
        # the mutated holdout eval_env) OR the construction scenario is itself paper.
        self._is_paper_regime = self.paper_regime or scenario in PAPER_SCENARIO_CONFIG
```

Replace `self.max_time = max_time` / `self.comfort_coeff = comfort_coeff` (`crowd_env.py:55-56`):

```python
        self.max_time = max_time if max_time is not None else (
            12.5 if self._is_paper_regime else 50.0)
        self.comfort_coeff = comfort_coeff if comfort_coeff is not None else (
            2.0 if self._is_paper_regime else 6.0)
```

Replace the `collision_threshold` block (`crowd_env.py:71-74`):

```python
        self.collision_threshold = (
            collision_threshold if collision_threshold is not None
            else (0.3 if self._is_paper_regime else self.robot_radius + self.human_radius)
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest test_paper_scenarios.py -k "regime or override" -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add crowd_sim/crowd_env.py test_paper_scenarios.py
git commit -m "v25: env paper-regime resolution (12.5s/comfort-2/d_col-0.3), non-paper unchanged"
```

---

### Task 2: Env 10m crossing geometry + update existing geometry tests

**Files:**
- Modify: `crowd_sim/crowd_env.py` (`reset()` paper branch `:175-184` → read from config, robot_y=5)
- Test: `test_paper_scenarios.py`

- [ ] **Step 1: Write/adjust the failing tests**

Update the two existing geometry tests and add a crossing/sense assertion in
`test_paper_scenarios.py`:

```python
def test_paper_standard_layout():
    env = CrowdSimEnv(num_humans=5, scenario='paper_standard', human_motion_model='orca')
    env.reset(seed=0)
    assert (env.robot_px, env.robot_py) == (0.0, -5.0)   # 10 m crossing (was -4)
    assert (env.robot_gx, env.robot_gy) == (0.0, 5.0)
    assert env.sense_range == 4.0
    half = 5.0  # 10x10 arena
    assert np.all(np.abs(env.humans_px) <= half) and np.all(np.abs(env.humans_py) <= half)
    radii = np.hypot(env.humans_px, env.humans_py)
    assert np.std(radii) > 0.3, f"humans look circle-placed: radii={radii}"
    d_start = np.hypot(env.humans_px - env.robot_px, env.humans_py - env.robot_py)
    assert np.all(d_start >= env.collision_threshold)
    assert env.human_vpref == 1.0


def test_paper_challenging_scales_arena():
    for n in (10, 15, 20):
        env = CrowdSimEnv(num_humans=n, scenario='paper_challenging', human_motion_model='orca')
        env.reset(seed=1)
        assert (env.robot_px, env.robot_py) == (0.0, -5.0)   # 10 m crossing (was -6)
        assert (env.robot_gx, env.robot_gy) == (0.0, 5.0)
        assert env.sense_range == 6.0
        half = 7.5  # 15x15 arena
        assert np.all(np.abs(env.humans_px) <= half) and np.all(np.abs(env.humans_py) <= half)
        assert env.humans_px.shape == (n,)


def test_paper_regime_easy_keeps_circle_geometry():
    # paper_regime forces the BUDGET but must NOT impose the paper crossing on 'easy'.
    env = CrowdSimEnv(num_humans=5, scenario='easy', human_motion_model='orca',
                      paper_regime=True)
    env.reset(seed=0)
    radii = np.hypot(env.humans_px, env.humans_py)
    assert np.allclose(radii, 4.0, atol=1e-9), f"easy geometry changed: radii={radii}"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest test_paper_scenarios.py -k "layout or scales_arena or circle_geometry" -v`
Expected: FAIL — robot still at ±4/±6; `sense_range` is None.

- [ ] **Step 3: Implement**

Replace the two paper `elif` branches in `reset()` (`crowd_env.py:175-184`) with one
config-driven branch:

```python
        elif self.scenario in PAPER_SCENARIO_CONFIG:
            cfg = PAPER_SCENARIO_CONFIG[self.scenario]
            self.human_vpref = 1.0  # parity with the 1.0 m/s robot
            scenario_type = 'paper'
            self._paper_arena = self.arena_size if self.arena_size is not None else cfg['arena']
            self._paper_robot_y = cfg['robot_y']
            if self.sense_range is None:
                self.sense_range = cfg['sense_range']
```

(The paper placement block at `crowd_env.py:247-279` already reads `self._paper_robot_y`,
so the 10 m crossing follows automatically.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest test_paper_scenarios.py -v`
Expected: PASS (all, including the unchanged `test_existing_hard_scenario_unchanged`).

- [ ] **Step 5: Commit**

```bash
git add crowd_sim/crowd_env.py test_paper_scenarios.py
git commit -m "v25: paper crossing 10m (robot +/-5) + sense 4/6 from config"
```

---

### Task 3: Thread `paper_regime` + None defaults through `train.py`

**Files:**
- Modify: `sncp_ppo/train.py` (`make_env` `:216-234`; main env `:274-284`; CLI `:1034`,`:1037`; `_train_vectorized` `:733-790`)
- Test: `test_paper_scenarios.py`

- [ ] **Step 1: Write the failing tests**

Add to `test_paper_scenarios.py`:

```python
def test_make_env_paper_regime_forces_budget():
    env = make_env(num_humans=3, scenario='easy', seed=0, paper_regime=True)()
    assert env.max_time == 12.5
    assert env.comfort_coeff == 2.0
    assert env.collision_threshold == 0.3


def test_make_env_nonpaper_defaults_unchanged():
    env = make_env(num_humans=5, scenario='hard', seed=0)()
    assert env.max_time == 50.0
    assert env.comfort_coeff == 6.0
    assert env.collision_threshold == env.robot_radius + env.human_radius


def test_train_parser_budget_defaults_are_none():
    args = build_parser().parse_args([])
    assert args.max_time is None
    assert args.comfort_coeff is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest test_paper_scenarios.py -k "make_env_paper or make_env_nonpaper or budget_defaults" -v`
Expected: FAIL — `make_env` has no `paper_regime`; parser defaults are 50.0/6.0.

- [ ] **Step 3: Implement**

In `sncp_ppo/train.py`, import the config (add to the existing `from crowd_sim.crowd_env import ...`):

```python
from crowd_sim.crowd_env import CrowdSimEnv, PAPER_SCENARIO_CONFIG
```

`make_env` (`:216-231`) — new defaults + flag, threaded to the constructor:

```python
def make_env(num_humans, scenario, seed, comfort_coeff=None, max_time=None,
             robot_vpref=0.26, human_vpref_override=None, human_goal_noise=0.0,
             human_motion_model='orca', collision_threshold=None, paper_regime=False):
    """Factory for a single CrowdSimEnv, used by SyncVectorEnv."""
    def _thunk():
        env = CrowdSimEnv(
            num_humans=num_humans,
            scenario=scenario,
            comfort_coeff=comfort_coeff,
            max_time=max_time,
            robot_vpref=robot_vpref,
            human_vpref_override=human_vpref_override,
            human_goal_noise=human_goal_noise,
            human_motion_model=human_motion_model,
            collision_threshold=collision_threshold,
            paper_regime=paper_regime,
        )
        env.reset(seed=seed)
        return env
    return _thunk
```

CLI defaults (`:1034`, `:1037`) → `None` (keep help text):

```python
    parser.add_argument('--comfort_coeff', type=float, default=None,
```
```python
    parser.add_argument('--max_time', type=float, default=None,
```

In `train()`, derive the flag once (right after `device = ...`, `:264-265`) and pass it to
the main env (`:274`):

```python
    paper_regime = getattr(args, 'fixed_scenario', None) in PAPER_SCENARIO_CONFIG
```
Add `paper_regime=paper_regime,` to the `CrowdSimEnv(...)` call at `:274-284`.

In `_train_vectorized` (`:732-790`), add the same derivation near the other `getattr`
reads (after `:741`):

```python
    paper_regime = getattr(args, 'fixed_scenario', None) in PAPER_SCENARIO_CONFIG
```
Add `paper_regime=paper_regime,` to the `make_env(...)` call inside `build_envs`
(`:750-761`) and to the `eval_env = CrowdSimEnv(...)` call (`:780-790`).

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest test_paper_scenarios.py -v`
Expected: PASS (all).

- [ ] **Step 5: Verify no arithmetic uses the now-None `args.max_time`**

Run: `python -c "import ast,sys; src=open('sncp_ppo/train.py').read(); print('max_time arithmetic check — inspect matches'); [print(l) for i,l in enumerate(src.splitlines(),1) if 'max_time' in l and ('/' in l or '*' in l or '+' in l or '-' in l) and 'self.max_time' not in l and 'env.max_time' not in l]"`
Expected: no lines printed (the only `max_time` arithmetic is on `env.max_time` at
`:108`, which reads the resolved value). If a line prints, change it to read
`env.max_time` after construction.

- [ ] **Step 6: Local smoke (tiny vectorized run, paper scenario, exit 0)**

Run:
```bash
python -m sncp_ppo.train --num_envs 2 --horizon 16 --total_steps 256 --eval_freq_updates 1 --fixed_scenario paper_challenging --num_humans 5 --robot_vpref 1.0 --holdout_scenarios paper_standard paper_challenging --holdout_episodes 1 --save_path checkpoints/sncp_ppo_v25_smoke.pt
```
Expected: exits 0; log shows the run starts and a holdout eval runs. Then confirm the
smoke env used the paper budget:
```bash
python -c "from crowd_sim.crowd_env import CrowdSimEnv; e=CrowdSimEnv(num_humans=5,scenario='easy',paper_regime=True); print('bootstrap max_time', e.max_time)"
```
Expected: `bootstrap max_time 12.5`. Delete the smoke checkpoint: `rm checkpoints/sncp_ppo_v25_smoke.pt`.

- [ ] **Step 7: Commit**

```bash
git add sncp_ppo/train.py test_paper_scenarios.py
git commit -m "v25: thread paper_regime (from --fixed_scenario) through train; budget defaults None"
```

---

### Task 4: Eval defaults to the paper budget for paper scenarios

**Files:**
- Modify: `sncp_ppo/eval_report.py` (`evaluate_density` `:434` default), `sncp_ppo/post_run_pipeline.py` (`:62`), `run_post_eval.py` (`:46`)
- Test: `test_paper_scenarios.py`

- [ ] **Step 1: Write the failing test**

Add to `test_paper_scenarios.py`:

```python
def test_eval_density_paper_uses_paper_budget(monkeypatch):
    # evaluate_density with max_time=None on a paper scenario must build a 12.5 s env.
    # evaluate_density builds the env (eval_report.py:451) BEFORE loading the checkpoint
    # (:456), so captured['max_time'] is set even if the checkpoint load later raises.
    import sncp_ppo.eval_report as er
    from crowd_sim.crowd_env import CrowdSimEnv as RealEnv
    captured = {}

    class _Spy(RealEnv):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            captured['max_time'] = self.max_time

    monkeypatch.setattr('crowd_sim.crowd_env.CrowdSimEnv', _Spy)
    try:
        er.evaluate_density(checkpoint_path='checkpoints/sncp_ppo_v24.pt',
                            num_humans=5, scenario='paper_challenging',
                            n_episodes=1, seed=0, robot_vpref=1.0,
                            human_vpref_override=1.0)  # max_time omitted -> None
    except Exception:
        pass  # we only need the env to have been constructed
    assert captured.get('max_time') == 12.5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest test_paper_scenarios.py -k "eval_density_paper" -v`
Expected: FAIL — `evaluate_density` default `max_time=50.0` → captured 50.0 (or paper checkpoint missing; if so, this test is environment-dependent — keep it but the GREEN signal is `captured['max_time'] == 12.5`).

- [ ] **Step 3: Implement**

`sncp_ppo/eval_report.py:434` — change `max_time: float = 50.0` → `max_time: float | None = None`.
(The body already passes `max_time=max_time` to `CrowdSimEnv` at `:454`, and reads
`env.max_time` for `max_steps` at `:463`, so a paper scenario resolves to 12.5.)

`sncp_ppo/post_run_pipeline.py:62` — change `max_time: float = 50.0` → `max_time: float | None = None`.

`run_post_eval.py:46-47` — change the arg default:

```python
    parser.add_argument("--max_time", type=float, default=None,
                        help="Episode time cap for eval; None lets paper scenarios use 12.5s.")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest test_paper_scenarios.py -k "eval_density_paper" -v`
Expected: PASS (`captured['max_time'] == 12.5`).

- [ ] **Step 5: Commit**

```bash
git add sncp_ppo/eval_report.py sncp_ppo/post_run_pipeline.py run_post_eval.py test_paper_scenarios.py
git commit -m "v25: eval max_time default None -> paper scenarios eval at 12.5s"
```

---

### Task 5: v25 notebook + guard test

**Files:**
- Modify: `sncp_ppo_colab.ipynb` (title markdown; training cell v24→v25; eval cell v24→v25)
- Test: `test_post_run_pipeline.py`

- [ ] **Step 1: Write the failing guard test**

Add to `test_post_run_pipeline.py` (follow the existing notebook-parsing helper there;
if none, load with `json.load(open('sncp_ppo_colab.ipynb', encoding='utf-8'))` and join
each code cell's `source`):

```python
def test_notebook_is_v25_paper_faithful():
    import json
    nb = json.load(open('sncp_ppo_colab.ipynb', encoding='utf-8'))
    code = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
    train_cells = [s for s in code if "sncp_ppo.train" in s and "--fixed_scenario" in s]
    eval_cells = [s for s in code if "run_post_eval.py" in s]
    assert len(train_cells) == 1 and len(eval_cells) == 1
    train, ev = train_cells[0], eval_cells[0]
    # Training: paper scenario + v25 save path; budget is env-derived (NOT on the CLI).
    assert "--fixed_scenario" in train and "paper_challenging" in train
    assert "checkpoints/sncp_ppo_v25.pt" in train
    # Eval: v25, paper baseline beeline 40 (10 m), and NO forced --max_time
    #       (the v24 failure: eval forced 15s while training ran 50s).
    assert "--version" in ev and "25" in ev
    assert "--baseline_nav_steps" in ev and "40" in ev
    assert "--max_time" not in ev
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest test_post_run_pipeline.py -k "v25_paper_faithful" -v`
Expected: FAIL — notebook still v24; eval cell still has `--max_time 15.0`.

- [ ] **Step 3: Implement (patch the notebook cells)**

Run this patch script (edits cells in place, preserving structure):

```python
import json
p = 'sncp_ppo_colab.ipynb'
nb = json.load(open(p, encoding='utf-8'))
def setsrc(cell, new): cell['source'] = new.splitlines(keepends=True)
for c in nb['cells']:
    if c['cell_type'] == 'markdown':
        s = ''.join(c['source'])
        if 'Current run' in s or 'v24' in s:
            setsrc(c, s.replace('v24', 'v25').replace(
                'paper-faithful scenario reproduction',
                'paper-faithful time budget + geometry (12.5s, d_col 0.3, 10m, comfort 2)'))
    if c['cell_type'] != 'code':
        continue
    s = ''.join(c['source'])
    if 'sncp_ppo.train' in s and '--fixed_scenario' in s:
        # v24 training cell already omits --max_time/--comfort_coeff/--collision_threshold;
        # the env now derives the 12.5s/comfort-2/d_col-0.3 paper regime from
        # --fixed_scenario paper_challenging. Only the version label changes.
        setsrc(c, s.replace('v24', 'v25'))
    if 'run_post_eval.py' in s:
        s = s.replace("'24'", "'25'").replace('v24', 'v25')
        s = s.replace("'--max_time', '15.0',\n", "")
        s = s.replace("'--baseline_nav_steps', '32'", "'--baseline_nav_steps', '40'")
        s = s.replace("'--nav_margin_steps', '8'", "'--nav_margin_steps', '10'")
        setsrc(c, s)
json.dump(nb, open(p, 'w', encoding='utf-8'), indent=1)
print('patched')
```

After running, open the training and eval cells and eyeball them (the assertions in
Step 1 are the real gate). If the eval cell lacks an explicit `--nav_margin_steps`, add
`'--nav_margin_steps', '10',` next to `--baseline_nav_steps`.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest test_post_run_pipeline.py -k "v25_paper_faithful" -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add sncp_ppo_colab.ipynb test_post_run_pipeline.py
git commit -m "v25: notebook to v25 (env-derived budget) + faithful-config guard test"
```

---

### Task 6: Full regression + parity check

**Files:** none (verification only)

- [ ] **Step 1: Run the full suite**

Run: `python -m pytest -q`
Expected: all pass. The previously-touched parity test
`test_vec_curriculum.py::test_holdout_config_is_parity_and_has_highdensity` still passes
(paper_* exempt at vpref 1.0; unchanged by this work).

- [ ] **Step 2: Behavioral parity spot-check (non-paper unchanged)**

Run:
```bash
python -c "from crowd_sim.crowd_env import CrowdSimEnv as E; a=E(num_humans=5,scenario='hard'); print(a.max_time,a.comfort_coeff,round(a.collision_threshold,2)); b=E(num_humans=5,scenario='circle'); print(b.max_time,b.comfort_coeff)"
```
Expected: `50.0 6.0 0.6` then `50.0 6.0` — the 0.26 m/s regime is untouched.

- [ ] **Step 3: Oracle feasibility sanity (10m crossing is learnable at 12.5s)**

Run:
```bash
python -c "
from crowd_sim.crowd_env import CrowdSimEnv
e=CrowdSimEnv(num_humans=10,scenario='paper_challenging')
e.reset(seed=0)
beeline=abs(e.robot_gy-e.robot_py)/(e.robot_vpref*e.time_step)
budget=int(e.max_time/e.time_step)+1
print('beeline %.0f steps, budget %d steps, margin %d' % (beeline, budget, budget-beeline))
"
```
Expected: `beeline 40 steps, budget 51 steps, margin 11` (≈2.5s of avoidance room — the
paper's tight-but-achievable regime).

- [ ] **Step 4: Commit (if any doc/notes updated)**

No code change expected here. If notes were updated, commit them; otherwise this task is
verification-only.

---

## Post-implementation (out of plan, operator/Colab)

After merge to `main`: open the notebook fresh from GitHub, run the v25 training cell
(2.5M, A100), download `eval_v25_artifacts.zip` + `sncp_ppo_v25.pt`, run
`python stage_colab_run_artifacts.py --version 25`, and compare the **12.5s** density
sweep to v24's 0% and to the paper's ~94% challenging. Update memory
(`sncp-paper-vs-impl.md`) with the actual v25 numbers.

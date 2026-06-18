# Repo cleanup & reorganization

Date: 2026-06-18
Branch: `chore/repo-cleanup`

## Goal

De-clutter the GitHub repo and the local working tree: move loose root-level code into
`tests/` and `scripts/`, stop tracking experiment-artifact binaries (checkpoints, old eval
dirs, result images), and expand `.gitignore` so new artifacts never re-clutter. **No behaviour
change to `sncp_ppo`/`crowd_sim`; the test suite, the run-readiness check, and the (training)
notebook path must all still pass/work.**

## Verified ground truth

- **README.md / docs/ do NOT reference any root result image** (`gif_v14_*`, `traj_v14/v15_*`,
  `v15_results.png`, `density_sweep_v14.png`) — untracking them breaks no rendered docs.
- **10 test files import root-level scripts at function scope** (so the scripts must stay
  importable after moving): `verify_v16_artifacts`, `evaluate_custom_scenario`,
  `evaluate_policy_report`, `compare_policy_reports`, `run_post_eval`, `run_v16_post_eval`,
  `analyze_training_log`, `run_v17_review`, `select_v18_candidate`, `verify_v18_ready`.
  → a root `conftest.py` must put repo-root + `scripts/` + `scripts/archive/` on `sys.path`.
- **Tests' root-relative reads** (`Path(".")`, `Path("sncp_ppo_colab.ipynb")`) resolve to the
  repo root because pytest does not change CWD — unaffected by moving test files into `tests/`.
- **Training is unaffected:** the notebook trains via `python -m sncp_ppo.train` (module, not a
  script path). Only the (already non-functional) eval/viz cells call scripts by path.

## Target structure

```
repo/
├── README.md AGENTS.md requirements.txt ruff.toml .gitignore conftest.py(NEW)
├── sncp_ppo_colab.ipynb            (stays at root — Colab opens it here)
├── sncp_ppo/ crowd_sim/            (packages — untouched)
├── custom_scenarios/ custom_map_app/ waffle_ros/ docs/ demo/   (untouched)
├── tests/                          (ALL test_*.py)
└── scripts/
    ├── <active tools>              run_post_eval, stage_colab_run_artifacts,
    │                               visualize_trajectory, visualize_trajectory_gif,
    │                               visualize_all_scenarios_gif, plot_training,
    │                               evaluate_policy_report, evaluate_custom_scenario,
    │                               compare_policy_reports, analyze_training_log,
    │                               benchmark, benchmark_orca, benchmark_ppo
    └── archive/ <version one-offs> run_v16_post_eval, run_v17_review, select_v18_candidate,
                                    verify_v16_artifacts, verify_v16_run_ready,
                                    verify_v18_ready, run_probes, _bench_sfm,
                                    visualize_architecture
```

`scratch/` (gitignored, NEW) ← the 7 untracked one-off probes (`_probe_attn.py`,
`_probe_il.py`, `_sweep_expert.py`, `_bench_endtoend.py`, `_eval_v24_corrected.py`,
`_make_crowd_gifs.py`, `_oracle_feasibility.py`) — moved out of root, kept local, never tracked.

## Disposition rules

1. **`conftest.py` (root, NEW):** inserts repo-root, `scripts/`, `scripts/archive/` into
   `sys.path` so tests' function-level `import <script>` keeps resolving after the move.
2. **`git mv` all `test_*.py` → `tests/`** (preserves history; `conftest.py` is the import
   bridge).
3. **`git mv` the loose scripts → `scripts/` (active) and `scripts/archive/` (version one-offs)**
   per the tree above.
4. **Notebook path updates:** the 4 script calls the notebook makes
   (`run_post_eval.py`, `visualize_trajectory.py`, `visualize_trajectory_gif.py`,
   `plot_training.py`) → `scripts/<name>.py`. (Run-readiness/version-marker tokens are
   substrings like `run_post_eval.py`, so they still match.)
5. **Untrack from GitHub (`git rm --cached`, local copies + history retained):**
   `checkpoints/*.pt` (9 milestones, 19M), `eval_v15/ eval_v16/ eval_v18/ eval_v19/ eval_v21/
   eval_v22/` (old eval artifacts), the root result images listed above.
6. **`.gitignore` additions:** `/eval_v*/`, `/*.zip`, `/training_*.csv`,
   `/*_multiseed_result.json`, `/sncp_ppo_v*.pt`, `checkpoints/*.pt`, `/*.png`, `/*.gif`
   (root-anchored — only repo-root images; `demo/` is a subdir and keeps its `!demo/*.gif`
   rule), `rapor/`, `ltc_sunum/`, `ltc_blog_app/`, `scratch/`.

## Verification (after each phase; full set at the end)

- `python -m pytest -q --basetemp=./.pytmp` → all currently-passing tests still pass
  (206 expected). This is the primary safety net (catches any broken import/path).
- Run-readiness preflight: `verify_v16_run_ready(Path("."))` → `pass` (notebook still v29-ready;
  the script-path edits keep the substring tokens valid).
- Notebook JSON still valid + a CLI smoke of one moved active script
  (`python scripts/run_post_eval.py --help` exits 0) to confirm package imports resolve from the
  new location.

## Out of scope / deferred

- Moving the artifact binaries themselves (checkpoints/, eval_vXX/, root `.pt`/`.zip`/`.csv`):
  they stay in place but become git-ignored — moving them risks the notebook/eval paths and the
  canonical `checkpoints/sncp_ppo_vXX.pt` references. The user can delete local copies manually.
- Restructuring `sncp_ppo/` or `crowd_sim/` internals (no need; they're already clean packages).
- Rewriting git history to purge the large binaries from past commits (history is retained;
  only future tracking changes).

## Irreversibility note

`git rm --cached` + the moves are committed and **pushed to main** only after the user confirms
at the finishing step. History is preserved; nothing is deleted from disk. The push changes the
repo's tracked file set going forward.

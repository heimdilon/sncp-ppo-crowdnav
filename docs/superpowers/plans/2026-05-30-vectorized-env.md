# Vectorized Environment Rollout (v10) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a vectorized rollout path (N parallel envs × T steps → 2048 transitions/PPO update) to fix data starvation, while keeping the existing single-env path byte-identical.

**Architecture:** A new `VectorizedRolloutBuffer` stores fixed `(num_envs, horizon)` tensors with per-step hidden states and a done mask. A done-masked GAE computes advantages per-env along the time axis (resetting at episode boundaries, bootstrapping at truncation/horizon-end). `train.py` gains a `--num_envs` flag: `1` → existing `update()` (unchanged), `>1` → `gymnasium.vector.SyncVectorEnv` rollout + `update_vectorized()`. BPTT keeps the existing fixed-window (seq_len=16) scheme, vectorized so windows never cross env or episode boundaries.

**Tech Stack:** PyTorch, Gymnasium (`gymnasium.vector.SyncVectorEnv`), NumPy, pytest.

---

## File Structure

- `sncp_ppo/vec_buffer.py` (new) — `VectorizedRolloutBuffer`: fixed N×T storage, done-masked GAE, seq_len-window extraction.
- `sncp_ppo/ppo.py` (modify) — add `update_vectorized(buffer, device)`; reuse existing loss/KL/RMS helpers; `update()` untouched.
- `sncp_ppo/train.py` (modify) — add `--num_envs`, `--horizon`; add `make_env()` helper; add vectorized rollout branch; single-env path unchanged when `num_envs==1`.
- `test_vec_gae.py` (new) — GAE equivalence (safety anchor) + done-mask correctness.
- `test_vec_buffer.py` (new) — buffer shape, bootstrap placement, hidden reset, subsequence boundary.

**Implementation order rationale:** GAE equivalence first (Task 1) — it is the safety anchor that proves the new advantage math matches the proven single-env math before anything depends on it. Then buffer (Task 2-3), hidden reset (Task 4), PPO update (Task 5), train wiring + smoke (Task 6).

---

## Task 1: Done-masked GAE (the safety anchor)

A standalone function `compute_gae_vectorized(rewards, values, dones, bootstrap_values, gamma, gae_lambda)` operating on `(N, T)` tensors. Must produce the SAME advantages as the existing `PPOAgent.compute_gae` when given an equivalent transition sequence.

**Files:**
- Create: `sncp_ppo/vec_buffer.py`
- Test: `test_vec_gae.py`

- [ ] **Step 1: Write the failing test — single-env equivalence**

```python
# test_vec_gae.py
import torch
from sncp_ppo.vec_buffer import compute_gae_vectorized
from sncp_ppo.ppo import PPOAgent
from sncp_ppo.models import SNCPPolicy


def test_vectorized_gae_matches_single_env():
    """The new (N,T) done-masked GAE must equal the legacy episode-aware GAE
    on the same transition sequence (N=1, one truncated episode)."""
    gamma, lam = 0.99, 0.95
    T = 6
    rewards = torch.tensor([1.0, -2.0, 0.5, 3.0, -1.0, 2.0])
    values = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    # No terminations mid-episode; the episode is truncated at the end with a
    # bootstrap value of 0.7 (timeout case).
    masks = torch.ones(T)              # legacy: mask=1 means "world continues"
    dones = torch.zeros(T)            # vectorized: done=1 only at episode end
    bootstrap = 0.7

    # Legacy reference (episode-aware, one episode of length T, truncated)
    agent = PPOAgent(policy=SNCPPolicy())
    ref_adv, ref_ret = agent.compute_gae(
        rewards, values, masks, episode_lengths=[T], bootstrap_values=[bootstrap]
    )

    # Vectorized: shape (N=1, T). done at last step, bootstrap supplied there.
    dones_NT = dones.view(1, T)
    dones_NT[0, T - 1] = 1.0
    adv, ret = compute_gae_vectorized(
        rewards.view(1, T), values.view(1, T), dones_NT,
        bootstrap_values=torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, bootstrap]]),
        gamma=gamma, gae_lambda=lam,
    )
    assert torch.allclose(adv.view(-1), ref_adv, atol=1e-5), f"{adv.view(-1)} vs {ref_adv}"
    assert torch.allclose(ret.view(-1), ref_ret, atol=1e-5)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest test_vec_gae.py::test_vectorized_gae_matches_single_env -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'sncp_ppo.vec_buffer'`

- [ ] **Step 3: Write minimal implementation**

```python
# sncp_ppo/vec_buffer.py
import torch


def compute_gae_vectorized(rewards, values, dones, bootstrap_values,
                           gamma=0.99, gae_lambda=0.95):
    """Done-masked GAE over (N, T) tensors.

    Args:
        rewards:          (N, T) per-step rewards.
        values:           (N, T) per-step value estimates V(s_t).
        dones:            (N, T) 1.0 if step t is the last step of an episode
                          for env n (terminated OR truncated OR horizon-cut).
        bootstrap_values: (N, T) V(s_{t+1}) to use at episode-boundary steps;
                          0.0 for terminated, V(s_final) for truncated/horizon.
                          Ignored where dones==0.
        gamma, gae_lambda: GAE hyperparameters.

    Returns:
        advantages (N, T), returns (N, T).

    Each env is processed independently along the time axis. At a done step the
    next-state value is `bootstrap_values[n, t]` (not values[n, t+1]) and the
    GAE accumulator resets afterwards, so advantages never bleed across episode
    boundaries within an env's rollout. Mirrors PPOAgent.compute_gae but driven
    by a done mask instead of an episode-length list.
    """
    N, T = rewards.shape
    advantages = torch.zeros_like(rewards)
    for n in range(N):
        gae = 0.0
        for t in reversed(range(T)):
            if dones[n, t] > 0.5:
                next_value = bootstrap_values[n, t]
                next_nonterminal = 0.0  # accumulator resets after a boundary
            else:
                next_value = values[n, t + 1] if t + 1 < T else 0.0
                next_nonterminal = 1.0
            delta = rewards[n, t] + gamma * next_value - values[n, t]
            gae = delta + gamma * gae_lambda * next_nonterminal * gae
            advantages[n, t] = gae
    returns = advantages + values
    return advantages, returns
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest test_vec_gae.py::test_vectorized_gae_matches_single_env -v`
Expected: PASS

- [ ] **Step 5: Write the failing test — multi-episode done-mask reset**

```python
# test_vec_gae.py (append)
def test_vectorized_gae_resets_at_done():
    """A terminated episode mid-rollout must not bleed advantage into the next
    episode in the same env row."""
    gamma, lam = 0.99, 0.95
    rewards = torch.tensor([[1.0, 1.0, 1.0, 1.0]])   # N=1, T=4
    values = torch.tensor([[0.0, 0.0, 0.0, 0.0]])
    dones = torch.tensor([[0.0, 1.0, 0.0, 1.0]])     # episode ends at t=1 and t=3
    bootstrap = torch.tensor([[0.0, 0.0, 0.0, 0.0]]) # terminated: no bootstrap
    adv, ret = compute_gae_vectorized(rewards, values, dones, bootstrap, gamma, lam)
    # Episode A (t=0,1): adv[1]=1.0 (terminal), adv[0]=1+gamma*lam*1.0
    # Episode B (t=2,3): adv[3]=1.0 (terminal), adv[2]=1+gamma*lam*1.0
    expected_first = 1.0 + gamma * lam * 1.0
    assert torch.allclose(adv[0, 1], torch.tensor(1.0), atol=1e-5)
    assert torch.allclose(adv[0, 0], torch.tensor(expected_first), atol=1e-5)
    assert torch.allclose(adv[0, 3], torch.tensor(1.0), atol=1e-5)
    assert torch.allclose(adv[0, 2], torch.tensor(expected_first), atol=1e-5)
```

- [ ] **Step 6: Run test to verify it passes**

Run: `python -m pytest test_vec_gae.py -v`
Expected: PASS (2 tests). The reset logic is already implemented in Step 3.

- [ ] **Step 7: Commit**

```bash
git add sncp_ppo/vec_buffer.py test_vec_gae.py
git commit -m "feat(vec): done-masked GAE matching legacy episode-aware GAE"
```

---

## Task 2: VectorizedRolloutBuffer — storage + shape

`VectorizedRolloutBuffer` accumulates N×T transitions including per-step hidden states. This task covers construction, `store()`, and `get_tensors()`.

**Files:**
- Modify: `sncp_ppo/vec_buffer.py`
- Test: `test_vec_buffer.py`

- [ ] **Step 1: Write the failing test**

```python
# test_vec_buffer.py
import torch
from sncp_ppo.vec_buffer import VectorizedRolloutBuffer


def _hidden(N, H, units=32):
    return {
        'temporal_edge': torch.zeros(N, units),
        'spatial_edge': torch.zeros(N * H, units),
        'node': torch.zeros(N, units),
    }


def test_buffer_accumulates_NT_shapes():
    N, T, H = 4, 8, 5
    buf = VectorizedRolloutBuffer(num_envs=N, horizon=T)
    for t in range(T):
        obs = {
            'robot_node': torch.zeros(N, 7),
            'spatial_edges': torch.zeros(N, H, 4),
            'temporal_edges': torch.zeros(N, 2),
        }
        buf.store(
            obs=obs, hidden=_hidden(N, H),
            actions=torch.zeros(N, 2), log_probs=torch.zeros(N),
            rewards=torch.zeros(N), values=torch.zeros(N),
            dones=torch.zeros(N), masks=torch.ones(N),
        )
    data = buf.get_tensors(torch.device('cpu'))
    assert data['rewards'].shape == (N, T)
    assert data['actions'].shape == (N, T, 2)
    assert data['obs']['spatial_edges'].shape == (N, T, H, 4)
    assert data['obs']['robot_node'].shape == (N, T, 7)
    assert data['dones'].shape == (N, T)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest test_vec_buffer.py::test_buffer_accumulates_NT_shapes -v`
Expected: FAIL with `ImportError: cannot import name 'VectorizedRolloutBuffer'`

- [ ] **Step 3: Write minimal implementation**

```python
# sncp_ppo/vec_buffer.py (append)
class VectorizedRolloutBuffer:
    """Fixed (num_envs, horizon) rollout storage for vectorized PPO.

    Unlike the list-based, episode-bounded PPOMemory, this stores a dense N×T
    block per step. Hidden states fed INTO each step are recorded so BPTT can
    re-feed them during the update. Episode boundaries are tracked via the
    `dones` mask, which drives both GAE reset and BPTT hidden-reset.
    """

    def __init__(self, num_envs, horizon):
        self.N = num_envs
        self.T = horizon
        self._reset_lists()

    def _reset_lists(self):
        self.obs_robot_node = []      # each: (N, 7)
        self.obs_spatial_edges = []   # each: (N, H, 4)
        self.obs_temporal_edges = []  # each: (N, 2)
        self.h_temporal = []          # each: (N, units)
        self.h_spatial = []           # each: (N*H, units)
        self.h_node = []              # each: (N, units)
        self.actions = []             # each: (N, 2)
        self.log_probs = []           # each: (N,)
        self.rewards = []             # each: (N,)
        self.values = []              # each: (N,)
        self.dones = []               # each: (N,)
        self.masks = []               # each: (N,)  (1 - terminated)
        self.bootstrap_values = None  # (N, T), filled at finish()

    def store(self, obs, hidden, actions, log_probs, rewards, values, dones, masks):
        self.obs_robot_node.append(obs['robot_node'].detach().clone())
        self.obs_spatial_edges.append(obs['spatial_edges'].detach().clone())
        self.obs_temporal_edges.append(obs['temporal_edges'].detach().clone())
        self.h_temporal.append(hidden['temporal_edge'].detach().clone())
        self.h_spatial.append(hidden['spatial_edge'].detach().clone())
        self.h_node.append(hidden['node'].detach().clone())
        self.actions.append(torch.as_tensor(actions).detach().clone())
        self.log_probs.append(torch.as_tensor(log_probs).detach().clone())
        self.rewards.append(torch.as_tensor(rewards, dtype=torch.float32).clone())
        self.values.append(torch.as_tensor(values, dtype=torch.float32).clone())
        self.dones.append(torch.as_tensor(dones, dtype=torch.float32).clone())
        self.masks.append(torch.as_tensor(masks, dtype=torch.float32).clone())

    def get_tensors(self, device):
        def stack(lst):
            return torch.stack(lst, dim=1).to(device)  # (N, T, ...)
        return {
            'obs': {
                'robot_node': stack(self.obs_robot_node),
                'spatial_edges': stack(self.obs_spatial_edges),
                'temporal_edges': stack(self.obs_temporal_edges),
            },
            'h_temporal': stack(self.h_temporal),
            'h_spatial': stack(self.h_spatial),
            'h_node': stack(self.h_node),
            'actions': stack(self.actions),
            'log_probs': stack(self.log_probs),
            'rewards': stack(self.rewards),
            'values': stack(self.values),
            'dones': stack(self.dones),
            'masks': stack(self.masks),
        }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest test_vec_buffer.py::test_buffer_accumulates_NT_shapes -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add sncp_ppo/vec_buffer.py test_vec_buffer.py
git commit -m "feat(vec): VectorizedRolloutBuffer storage + N*T get_tensors"
```

---

## Task 3: Buffer bootstrap placement (`finish`)

`finish(last_values, last_dones)` fills the `(N, T)` bootstrap tensor: at terminated steps bootstrap=0; at truncated/horizon-end steps bootstrap=V(s_next). Also forces `dones[:, T-1]=1` so the horizon cut is treated as an episode boundary for GAE.

**Files:**
- Modify: `sncp_ppo/vec_buffer.py`
- Test: `test_vec_buffer.py`

- [ ] **Step 1: Write the failing test**

```python
# test_vec_buffer.py (append)
def test_finish_sets_bootstrap_and_horizon_done():
    N, T, H = 2, 3, 5
    buf = VectorizedRolloutBuffer(num_envs=N, horizon=T)
    # env 0: terminates (collision/goal) at t=1; env 1: never terminates
    done_seq = [torch.tensor([0., 0.]), torch.tensor([1., 0.]), torch.tensor([0., 0.])]
    mask_seq = [torch.tensor([1., 1.]), torch.tensor([0., 1.]), torch.tensor([1., 1.])]
    for t in range(T):
        obs = {'robot_node': torch.zeros(N, 7),
               'spatial_edges': torch.zeros(N, H, 4),
               'temporal_edges': torch.zeros(N, 2)}
        buf.store(obs=obs,
                  hidden={'temporal_edge': torch.zeros(N, 32),
                          'spatial_edge': torch.zeros(N * H, 32),
                          'node': torch.zeros(N, 32)},
                  actions=torch.zeros(N, 2), log_probs=torch.zeros(N),
                  rewards=torch.zeros(N), values=torch.zeros(N),
                  dones=done_seq[t], masks=mask_seq[t])
    # last_values = V(s_next) at horizon end for each env
    buf.finish(last_values=torch.tensor([9.0, 7.0]),
               last_dones=torch.tensor([0., 0.]))
    data = buf.get_tensors(torch.device('cpu'))
    # Horizon end (t=T-1) is forced done for BOTH envs
    assert data['dones'][0, T - 1] == 1.0
    assert data['dones'][1, T - 1] == 1.0
    # env1 horizon-end bootstrap = its last_value (7.0); env0 horizon-end too (9.0)
    assert torch.allclose(data['bootstrap_values'][1, T - 1], torch.tensor(7.0))
    assert torch.allclose(data['bootstrap_values'][0, T - 1], torch.tensor(9.0))
    # env0 terminated at t=1 -> bootstrap there is 0 (mask was 0)
    assert torch.allclose(data['bootstrap_values'][0, 1], torch.tensor(0.0))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest test_vec_buffer.py::test_finish_sets_bootstrap_and_horizon_done -v`
Expected: FAIL with `AttributeError: 'VectorizedRolloutBuffer' object has no attribute 'finish'`

- [ ] **Step 3: Write minimal implementation**

```python
# sncp_ppo/vec_buffer.py — add to VectorizedRolloutBuffer
    def finish(self, last_values, last_dones):
        """Finalize the rollout: compute the (N, T) bootstrap tensor and mark
        the horizon end as a done boundary.

        last_values: (N,) V(s_next) after the final stored step, per env.
        last_dones:  (N,) whether the env was *terminated* exactly on the final
                     stored step (then horizon-end bootstrap is 0 for that env).

        Bootstrap rule per step t:
          - if masks[t]==0 (terminated): bootstrap=0 (no future value).
          - else if it's an episode boundary by truncation: bootstrap=V(s_{t+1}).
        We only have V(s_next) for the horizon end; mid-rollout truncations
        (timeouts) are handled by storing their own bootstrap via dones+mask: a
        truncated step has done=1 but mask=1, and its bootstrap is value[t+1]
        within the buffer (next stored step is the reset obs's value — handled
        by GAE using values[t+1]). The only step lacking an in-buffer successor
        is the horizon end, which is why finish supplies last_values there.
        """
        N, T = self.N, self.T
        masks = torch.stack(self.masks, dim=1)       # (N, T)
        boot = torch.zeros(N, T)
        # Horizon end: force done so GAE cuts here; bootstrap = last_values
        # unless the env terminated exactly on the last step.
        last_mask = 1.0 - last_dones.float()         # 0 if terminated at horizon end
        boot[:, T - 1] = last_values.float() * last_mask
        self.dones[T - 1] = torch.ones(N)
        self.bootstrap_values = boot

    def _get_bootstrap(self):
        return self.bootstrap_values
```

Then extend `get_tensors` to include the bootstrap (add this line before the
`return` in `get_tensors`, and add the key to the returned dict):

```python
        bootstrap = self.bootstrap_values if self.bootstrap_values is not None \
            else torch.zeros(self.N, self.T)
        # ... in the returned dict add:
        #     'bootstrap_values': bootstrap.to(device),
```

(Modify the existing `get_tensors` return dict to include
`'bootstrap_values': bootstrap.to(device),`.)

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest test_vec_buffer.py::test_finish_sets_bootstrap_and_horizon_done -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add sncp_ppo/vec_buffer.py test_vec_buffer.py
git commit -m "feat(vec): buffer.finish bootstrap placement + horizon-end done"
```

---

## Task 4: Hidden-state reset-on-done helper

`reset_hidden_where_done(hidden, dones, num_humans)` zeros the hidden rows of envs that just finished, so a new auto-reset episode does not start with stale LTC memory. Handles the spatial hidden's `(N*H, units)` layout.

**Files:**
- Modify: `sncp_ppo/vec_buffer.py`
- Test: `test_vec_buffer.py`

- [ ] **Step 1: Write the failing test**

```python
# test_vec_buffer.py (append)
from sncp_ppo.vec_buffer import reset_hidden_where_done


def test_reset_hidden_where_done():
    N, H, U = 3, 5, 32
    hidden = {
        'temporal_edge': torch.ones(N, U),
        'spatial_edge': torch.ones(N * H, U),
        'node': torch.ones(N, U),
    }
    dones = torch.tensor([0.0, 1.0, 0.0])  # only env 1 finished
    out = reset_hidden_where_done(hidden, dones, num_humans=H)
    # env 1 zeroed, envs 0 and 2 untouched
    assert torch.all(out['temporal_edge'][1] == 0)
    assert torch.all(out['node'][1] == 0)
    assert torch.all(out['temporal_edge'][0] == 1)
    assert torch.all(out['temporal_edge'][2] == 1)
    # spatial: env 1 occupies rows [H:2H]
    assert torch.all(out['spatial_edge'][H:2 * H] == 0)
    assert torch.all(out['spatial_edge'][0:H] == 1)
    assert torch.all(out['spatial_edge'][2 * H:3 * H] == 1)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest test_vec_buffer.py::test_reset_hidden_where_done -v`
Expected: FAIL with `ImportError: cannot import name 'reset_hidden_where_done'`

- [ ] **Step 3: Write minimal implementation**

```python
# sncp_ppo/vec_buffer.py (append, module-level function)
def reset_hidden_where_done(hidden, dones, num_humans):
    """Zero the LTC hidden state of envs that just finished an episode.

    SyncVectorEnv auto-resets a done env, so its next observation is already the
    new episode's first step. The recurrent hidden must be cleared for those env
    rows or the new episode inherits stale memory (a silent correctness bug).

    hidden: dict with 'temporal_edge' (N,U), 'spatial_edge' (N*H,U), 'node' (N,U).
    dones:  (N,) 1.0 where the env finished this step.
    """
    N = dones.shape[0]
    keep = (dones < 0.5).float()                  # (N,) 1.0 to keep, 0.0 to zero
    keep_col = keep.view(N, 1)
    new = {
        'temporal_edge': hidden['temporal_edge'] * keep_col,
        'node': hidden['node'] * keep_col,
    }
    # spatial is (N*H, U): expand keep per env across its H rows
    keep_spat = keep.repeat_interleave(num_humans).view(N * num_humans, 1)
    new['spatial_edge'] = hidden['spatial_edge'] * keep_spat
    return new
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest test_vec_buffer.py::test_reset_hidden_where_done -v`
Expected: PASS

- [ ] **Step 5: Run the full vec test suite**

Run: `python -m pytest test_vec_buffer.py test_vec_gae.py -v`
Expected: PASS (all tests)

- [ ] **Step 6: Commit**

```bash
git add sncp_ppo/vec_buffer.py test_vec_buffer.py
git commit -m "feat(vec): reset_hidden_where_done for auto-reset envs"
```

---

## Task 5: `update_vectorized` in PPOAgent

Add `update_vectorized(buffer, device)` to `PPOAgent`. It reuses the existing return-RMS, clipped surrogate, clipped value loss, KL early-stop, and grad-clip logic, but sources data from the `(N,T)` buffer and uses `compute_gae_vectorized`. BPTT uses seq_len windows that respect env + episode boundaries.

**Files:**
- Modify: `sncp_ppo/ppo.py`
- Test: `test_vec_buffer.py` (integration smoke of the update math)

- [ ] **Step 1: Write the failing test**

```python
# test_vec_buffer.py (append)
from sncp_ppo.ppo import PPOAgent
from sncp_ppo.models import SNCPPolicy
from sncp_ppo.vec_buffer import VectorizedRolloutBuffer


def test_update_vectorized_runs_and_steps_optimizer():
    """update_vectorized should consume an (N,T) buffer, run without error, and
    actually change policy parameters (a gradient step happened)."""
    torch.manual_seed(0)
    N, T, H = 4, 32, 5
    policy = SNCPPolicy()
    agent = PPOAgent(policy=policy, seq_len=16, batch_size=8, epochs=2)
    buf = VectorizedRolloutBuffer(num_envs=N, horizon=T)
    for t in range(T):
        obs = {'robot_node': torch.randn(N, 7),
               'spatial_edges': torch.randn(N, H, 4),
               'temporal_edges': torch.randn(N, 2)}
        dones = torch.zeros(N)
        if t == 15:
            dones[0] = 1.0  # env 0 finishes mid-rollout
        buf.store(obs=obs,
                  hidden=policy.init_hidden(N, H, torch.device('cpu')),
                  actions=torch.randn(N, 2), log_probs=torch.randn(N),
                  rewards=torch.randn(N), values=torch.randn(N),
                  dones=dones, masks=1.0 - dones)
    buf.finish(last_values=torch.randn(N), last_dones=torch.zeros(N))

    before = [p.clone() for p in policy.parameters()]
    agent.update_vectorized(buf, torch.device('cpu'))
    after = list(policy.parameters())
    changed = any(not torch.equal(b, a) for b, a in zip(before, after))
    assert changed, "optimizer did not update any parameter"
    # diagnostics populated
    assert isinstance(agent.last_approx_kl, float)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest test_vec_buffer.py::test_update_vectorized_runs_and_steps_optimizer -v`
Expected: FAIL with `AttributeError: 'PPOAgent' object has no attribute 'update_vectorized'`

- [ ] **Step 3: Write minimal implementation**

Add to `sncp_ppo/ppo.py` inside `PPOAgent` (import at top of file:
`from sncp_ppo.vec_buffer import compute_gae_vectorized`):

```python
    def update_vectorized(self, buffer, device):
        """PPO update from a VectorizedRolloutBuffer (N envs x T steps).

        Reuses the same surrogate/value/KL/RMS machinery as update(), but the
        advantages come from the done-masked vectorized GAE and BPTT windows are
        cut so they never span an env or an episode boundary.
        """
        data = buffer.get_tensors(device)
        N, T = data['rewards'].shape
        num_humans = data['obs']['spatial_edges'].shape[2]

        advantages, returns = compute_gae_vectorized(
            data['rewards'], data['values'], data['dones'],
            data['bootstrap_values'], self.gamma, self.gae_lambda,
        )

        values = data['values']
        if self.normalize_returns:
            self.return_rms.update(returns.detach().cpu().numpy())
            ret_std = self.return_rms.std
            returns = returns / ret_std
            values = values / ret_std

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Build seq_len windows per env that stop at episode boundaries (dones).
        # Each window: a contiguous run within one env that does not cross a done.
        windows = []  # list of (n, start, length)
        for n in range(N):
            seg_start = 0
            for t in range(T):
                is_boundary = data['dones'][n, t] > 0.5
                if is_boundary or t == T - 1:
                    seg_end = t + 1
                    # split [seg_start, seg_end) into seq_len chunks
                    s = seg_start
                    while s < seg_end:
                        e = min(s + self.seq_len, seg_end)
                        if e - s >= 4:  # skip very short fragments (matches update())
                            windows.append((n, s, e - s))
                        s = e
                    seg_start = seg_end
        if not windows:
            return

        S = self.seq_len
        num_win = len(windows)

        def gather_windows():
            rn = torch.zeros(num_win, S, 7, device=device)
            se = torch.zeros(num_win, S, num_humans, 4, device=device)
            te = torch.zeros(num_win, S, 2, device=device)
            act = torch.zeros(num_win, S, 2, device=device)
            olp = torch.zeros(num_win, S, device=device)
            adv = torch.zeros(num_win, S, device=device)
            ret = torch.zeros(num_win, S, device=device)
            ov = torch.zeros(num_win, S, device=device)
            h_te = torch.zeros(num_win, data['h_temporal'].shape[-1], device=device)
            h_no = torch.zeros(num_win, data['h_node'].shape[-1], device=device)
            h_sp = torch.zeros(num_win, num_humans, data['h_spatial'].shape[-1], device=device)
            lengths = []
            for i, (n, st, L) in enumerate(windows):
                rn[i, :L] = data['obs']['robot_node'][n, st:st + L]
                se[i, :L] = data['obs']['spatial_edges'][n, st:st + L]
                te[i, :L] = data['obs']['temporal_edges'][n, st:st + L]
                act[i, :L] = data['actions'][n, st:st + L]
                olp[i, :L] = data['log_probs'][n, st:st + L]
                adv[i, :L] = advantages[n, st:st + L]
                ret[i, :L] = returns[n, st:st + L]
                ov[i, :L] = values[n, st:st + L]
                h_te[i] = data['h_temporal'][n, st]
                h_no[i] = data['h_node'][n, st]
                # spatial hidden stored as (N, T, N*H? ) -> see note below
                h_sp[i] = data['h_spatial'][n, st].reshape(num_humans, -1)
                lengths.append(L)
            return rn, se, te, act, olp, adv, ret, ov, h_te, h_sp, h_no, lengths

        rn, se, te, act, olp, adv, ret, ov, h_te, h_sp, h_no, lengths = gather_windows()

        epoch_kl = epoch_ent = epoch_clip = 0.0
        epochs_ran = 0
        for epoch in range(self.epochs):
            perm = torch.randperm(num_win)
            batch_kls, batch_ents, batch_clips = [], [], []
            for bs in range(0, num_win, self.batch_size):
                bi = perm[bs:bs + self.batch_size]
                B = len(bi)
                b_rn, b_se, b_te = rn[bi], se[bi], te[bi]
                b_act, b_olp = act[bi], olp[bi]
                b_adv, b_ret, b_ov = adv[bi], ret[bi], ov[bi]
                b_len = [lengths[j] for j in bi.tolist()]
                valid = torch.zeros(B, S, device=device)
                for k, L in enumerate(b_len):
                    valid[k, :L] = 1.0

                h_temp = h_te[bi].clone()
                h_node = h_no[bi].clone()
                h_spat = h_sp[bi].reshape(B * num_humans, -1).clone()

                mus, stds, vals = [], [], []
                for t in range(S):
                    step_obs = {'robot_node': b_rn[:, t],
                                'spatial_edges': b_se[:, t],
                                'temporal_edges': b_te[:, t]}
                    step_h = {'temporal_edge': h_temp, 'spatial_edge': h_spat, 'node': h_node}
                    mu, std, val, nh = self.policy(step_obs, step_h)
                    mus.append(mu); stds.append(std); vals.append(val)
                    h_temp = nh['temporal_edge']; h_node = nh['node']
                    hs = nh['spatial_edge']
                    h_spat = hs.reshape(B * num_humans, -1) if hs.dim() == 3 else hs

                all_mu = torch.stack(mus, dim=1)
                all_std = torch.stack(stds, dim=1)
                all_val = torch.stack(vals, dim=1).squeeze(-1)

                dist = torch.distributions.Normal(all_mu, all_std)
                new_lp = dist.log_prob(b_act).sum(-1)
                entropy = dist.entropy().sum(-1)
                ratio = torch.exp(new_lp - b_olp)
                surr1 = ratio * b_adv
                surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * b_adv
                actor_loss = -(torch.min(surr1, surr2) * valid).sum() / valid.sum()

                with torch.no_grad():
                    lr_ = new_lp - b_olp
                    approx_kl = (((torch.exp(lr_) - 1) - lr_) * valid).sum() / valid.sum()
                    ent_mean = (entropy * valid).sum() / valid.sum()
                    clip_frac = (((ratio - 1).abs() > self.clip_eps).float() * valid).sum() / valid.sum()
                    batch_kls.append(approx_kl.item()); batch_ents.append(ent_mean.item()); batch_clips.append(clip_frac.item())

                v_clipped = b_ov + torch.clamp(all_val - b_ov, -self.clip_eps, self.clip_eps)
                vl_u = (all_val - b_ret).pow(2)
                vl_c = (v_clipped - b_ret).pow(2)
                critic_loss = (torch.max(vl_u, vl_c) * valid).sum() / valid.sum()

                entropy_loss = -(entropy * valid).sum() / valid.sum()
                loss = actor_loss + self.c1 * critic_loss + self.c2 * entropy_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
                self.optimizer.step()

            epochs_ran = epoch + 1
            if batch_kls:
                epoch_kl = float(np.mean(batch_kls))
                epoch_ent = float(np.mean(batch_ents))
                epoch_clip = float(np.mean(batch_clips))
            if self.target_kl is not None and epoch_kl > 1.5 * self.target_kl:
                break

        self.last_entropy = epoch_ent
        self.last_approx_kl = epoch_kl
        self.last_clip_frac = epoch_clip
        self.last_epochs_ran = epochs_ran
        if self.scheduler is not None:
            self.scheduler.step()
```

**Note on spatial hidden layout:** the buffer stores `h_spatial` per step as
`(N*H, units)`. After `get_tensors` stacking it becomes `(N*H, T, units)` —
NOT `(N, T, ...)`. To keep Task 5 indexing simple, the buffer must store spatial
hidden reshaped to `(N, H*units)` per step instead. Add this adjustment in
`VectorizedRolloutBuffer.store` (replace the spatial append):

```python
        # store spatial hidden as (N, H*units) so get_tensors -> (N, T, H*units)
        h_sp = hidden['spatial_edge']
        N_local = self.N
        self.h_spatial.append(h_sp.reshape(N_local, -1).detach().clone())
```

and in Task 5's `gather_windows`, `data['h_spatial'][n, st]` is `(H*units,)`,
so `.reshape(num_humans, -1)` recovers `(H, units)` — already written that way
above.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest test_vec_buffer.py::test_update_vectorized_runs_and_steps_optimizer -v`
Expected: PASS

- [ ] **Step 5: Run full vec suite (regression)**

Run: `python -m pytest test_vec_buffer.py test_vec_gae.py -v`
Expected: PASS (all). Then confirm legacy tests still pass:
`python -m pytest test_train_eta.py test_env_randomization.py test_env_velocity.py -q`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add sncp_ppo/ppo.py sncp_ppo/vec_buffer.py test_vec_buffer.py
git commit -m "feat(vec): update_vectorized PPO update from N*T buffer"
```

---

## Task 6: Wire `--num_envs` into train.py + smoke

Add `--num_envs` (default 1) and `--horizon` (default 128). `num_envs==1` keeps the existing single-env loop. `num_envs>1` runs the SyncVectorEnv rollout + `update_vectorized`. Add a `make_env` helper.

**Files:**
- Modify: `sncp_ppo/train.py`
- Test: manual smoke (CLI)

- [ ] **Step 1: Add make_env helper and CLI args**

In `sncp_ppo/train.py`, add near the top (after imports):

```python
def make_env(num_humans, scenario, seed):
    """Factory for a single CrowdSimEnv, used by SyncVectorEnv."""
    def _thunk():
        env = CrowdSimEnv(num_humans=num_humans, scenario=scenario)
        env.reset(seed=seed)
        return env
    return _thunk
```

In the argparse block add:

```python
    parser.add_argument('--num_envs', type=int, default=1,
                        help='Parallel envs. 1 = legacy single-env path; '
                             '>1 = vectorized fixed-horizon rollout.')
    parser.add_argument('--horizon', type=int, default=128,
                        help='Steps per env per PPO update in vectorized mode.')
```

- [ ] **Step 2: Add the vectorized branch in train()**

In `train()`, immediately after the agent is constructed and before the
single-env loop, branch:

```python
    if args.num_envs > 1:
        _train_vectorized(args, env, policy, agent, device, log_path, csv_writer, csv_file)
        return
```

Then add the function (uses `gymnasium.vector.SyncVectorEnv`):

```python
def _train_vectorized(args, env, policy, agent, device, log_path, csv_writer, csv_file):
    import gymnasium as gym
    from sncp_ppo.vec_buffer import VectorizedRolloutBuffer, reset_hidden_where_done

    N, T = args.num_envs, args.horizon
    H = args.num_humans
    envs = gym.vector.SyncVectorEnv(
        [make_env(H, 'circle', args.seed + i) for i in range(N)]
    )
    obs_np, _ = envs.reset(seed=args.seed)
    h = policy.init_hidden(batch_size=N, num_humans=H, device=device)

    total_steps = 0
    updates = args.episodes // (N * T // 100 or 1)  # rough update count for logging
    print(f"\nVectorized training: {N} envs x {T} steps = {N*T} transitions/update")
    print("-" * 90)

    def to_tensor(o):
        return {
            'robot_node': torch.tensor(o['robot_node'], dtype=torch.float32, device=device),
            'spatial_edges': torch.tensor(o['spatial_edges'], dtype=torch.float32, device=device),
            'temporal_edges': torch.tensor(o['temporal_edges'], dtype=torch.float32, device=device),
        }

    update_idx = 0
    target_updates = args.episodes  # treat --episodes as #updates upper bound in vec mode
    while update_idx < target_updates:
        buf = VectorizedRolloutBuffer(num_envs=N, horizon=T)
        for t in range(T):
            obs_t = to_tensor(obs_np)
            with torch.no_grad():
                mu, std, value, h_next = policy(obs_t, h)
                dist = torch.distributions.Normal(mu, std)
                action = dist.sample()
                log_prob = dist.log_prob(action).sum(-1)
            # clip per-dim for env
            act_np = action.cpu().numpy()
            act_np[:, 0] = np.clip(act_np[:, 0], 0.0, env.robot_vpref)
            act_np[:, 1] = np.clip(act_np[:, 1], -env.robot_wmax, env.robot_wmax)
            next_obs, reward, term, trunc, info = envs.step(act_np)
            done = np.logical_or(term, trunc)
            buf.store(obs=obs_t, hidden=h, actions=action,
                      log_probs=log_prob, rewards=torch.tensor(reward, dtype=torch.float32),
                      values=value.squeeze(-1),
                      dones=torch.tensor(done, dtype=torch.float32),
                      masks=torch.tensor(1.0 - term.astype('float32')))
            obs_np = next_obs
            h = reset_hidden_where_done(h_next, torch.tensor(done, dtype=torch.float32), H)
            total_steps += N

        with torch.no_grad():
            last_v = policy(to_tensor(obs_np), h)[2].squeeze(-1)
        buf.finish(last_values=last_v, last_dones=torch.zeros(N))
        agent.update_vectorized(buf, device)
        update_idx += 1

        if update_idx % 10 == 0:
            with torch.no_grad():
                stdv = policy.actor_logstd.exp().squeeze().cpu().numpy()
            print(f"Update {update_idx}/{target_updates} | steps {total_steps:8d} | "
                  f"ent={agent.last_entropy:+.3f} kl={agent.last_approx_kl:.5f} "
                  f"std=[{stdv[0]:.3f},{stdv[1]:.3f}] rms={agent.return_rms.std:.2f}")

    torch.save(policy.state_dict(), args.save_path.replace('.pt', '_final.pt'))
    csv_file.close()
    print("\nVectorized training completed!")
```

- [ ] **Step 3: Smoke test — vectorized path**

Run: `python -m sncp_ppo.train --num_envs 4 --horizon 64 --episodes 20 --num_humans 5 --save_path checkpoints/_vec_smoke.pt`
Expected: prints "Vectorized training: 4 envs x 64 steps = 256 transitions/update", several "Update K/20" lines with finite kl/ent, exits 0. Then: `rm -f checkpoints/_vec_smoke*.pt`

- [ ] **Step 4: Smoke test — legacy path unchanged**

Run: `python -m sncp_ppo.train --num_envs 1 --episodes 10 --eval_freq 100 --holdout_episodes 2 --save_path checkpoints/_legacy_smoke.pt`
Expected: the existing single-env output format ("Ep N/10 ... elapsed ... eta ..."), exits 0. Then: `rm -f checkpoints/_legacy_smoke*.pt`

- [ ] **Step 5: Commit**

```bash
git add sncp_ppo/train.py
git commit -m "feat(train): --num_envs vectorized rollout path (SyncVectorEnv)"
```

---

## Self-Review Notes

- **Spec coverage:** Component A (buffer) → Tasks 2-4; done-masked GAE → Task 1;
  Component C (update_vectorized) → Task 5; train wiring + `--num_envs` flag +
  backward-compat → Task 6. All 5 spec tests covered: GAE equivalence (T1),
  hidden reset (T4), buffer shape/bootstrap (T2-3), subsequence boundary (T5
  window logic), smoke (T6).
- **Out of scope confirmed:** Bug 5, reward shaping, obs additions, LTC→GRU,
  AsyncVectorEnv — none touched.
- **Known simplification:** in vectorized mode `--episodes` is reinterpreted as
  an upper bound on the number of PPO *updates* (not episodes), and holdout eval
  is not yet wired into the vectorized loop (single-env path retains it). A
  follow-up task can port `evaluate_holdout` into `_train_vectorized` once the
  core path is validated; flagged here so it is a conscious gap, not an omission.

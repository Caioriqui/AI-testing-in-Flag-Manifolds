"""
env.py -- RL environment for Objective 1 (MM845): building spherical
Coxeter diagrams by single-edge edits.

State and padding
------------------
The RL model has a FIXED maximum size N_MAX = 8 (project-wide, Diretrizes.txt),
but individual episodes act on a diagram with a variable number of vertices
`n_active <= N_MAX`, sampled per episode (see `active_ns` below / the
curriculum driven from train_ppo.py). This lets the same batched policy
network and the same fixed-size PPO rollout buffers handle every episode
size at once, instead of re-deriving shapes per N.

Concretely, the state is always an (N_MAX, N_MAX) int64 array in exactly
coxeter.py's representation, but only the top-left (n_active, n_active)
block is ever a "real" diagram; every entry touching an index >= n_active
is permanently NO_EDGE (padding vertices are always isolated, from both
each other and from the real vertices) and is never written to by `step`
(see the masking in policy.py, which the environment relies on to never
receive an action naming an invalid pair).

`n_active` cannot be recovered from the raw (N_MAX, N_MAX) array alone --
a real vertex that happens to have no edges at all is indistinguishable
from a padding vertex by construction. It is therefore threaded explicitly
alongside the diagram everywhere: `reset`/`step` return it, `StepInfo`
carries it, and policy.py/train_ppo.py pass it into every network call so
the message-passing and the action mask can be built correctly.

Why padding-with-isolated-vertices needs no special-casing in the reward
------------------------------------------------------------------------
The Cartan-type matrix of a diagram is A = 2*I - N, with N symmetric and
entrywise >= 0 (off-diagonal entries are -2*cos(pi/m) <= 0, so in N they
are >= 0). By Perron-Frobenius, a nonnegative symmetric matrix has a
nonnegative largest eigenvalue, so A's smallest eigenvalue is always
<= 2 -- with equality iff N = 0, i.e. the diagram has no edges at all.
A block of isolated padding vertices contributes only eigenvalues equal
to exactly 2 to the spectrum of the padded (N_MAX, N_MAX) Cartan matrix,
and those can therefore never be (strictly) the minimum unless the real
sub-diagram is ALSO totally disconnected, in which case its own margin
is also exactly 2. Either way:

    spherical_margin(padded) == spherical_margin(real n_active submatrix)

always, exactly (checked numerically in the sanity checks below). So
`step()` below keeps calling `spherical_margin_and_test` on the FULL
(N_MAX, N_MAX) state, with no slicing -- the reward and success signal are
already correct as long as padding stays isolated. Slicing to the real
(n_active, n_active) submatrix is only needed downstream for *reporting*
(analyze.py's Coxeter-type classification), where a real isolated vertex
and a padding vertex WOULD be conflated if you didn't slice first.

Action space
------------
A single discrete action combines "which unordered vertex pair" and "which
label to write there" (Diretrizes.txt explicitly rejects factoring this
into two sequential decisions as unneeded complexity at this scale). The
action space is sized for N_MAX regardless of the current episode's
n_active -- C(8, 2) = 28 pairs and 4 labels, so `NUM_ACTIONS = 28 * 4 =
112`, always -- and policy.py masks out (logit -inf) every action whose
pair touches an index >= n_active before sampling. This is the standard
way to handle a variable-size combinatorial action space inside a batched,
vectorised PPO loop without giving every episode a differently-shaped
tensor (which would break exactly the batching that makes training fast).
The encoding

    action = pair_index * len(LABELS) + label_index

is the single source of truth for this mapping (`encode_action` /
`decode_action` below); policy.py imports it from here rather than
re-deriving it, so the network's output ordering and the environment's
action semantics can never drift apart.

Episode
-------
Each episode starts from a diagram sampled i.i.d. by datagen.py (Dataset A
or B -- configurable) on `n_active` vertices, never from a corrupted
known-valid target, so the agent never gets to reverse-engineer a target
it was implicitly shown (Diretrizes.txt, and the project's stated
data-generation rationale). `n_active` is drawn uniformly from
`self.active_ns` at every `reset()` -- a tuple that train_ppo.py mutates
in place (via `VecDynkinEnv.set_active_ns`) to drive the curriculum over
N described in train_ppo.py, without needing to recreate any environment.
At every step the agent overwrites one entry M[i, j] = M[j, i] with the
label its action names -- including a "no-op" of writing back the label
already there. An episode ends when the diagram becomes spherical
(success) or a fixed step budget is exhausted (truncation).

Reward
------
Dense reward at every step: `spherical_margin` of the state AFTER the
action, squashed through tanh to keep it inside (-1, 1) and bound the
variance of the PPO advantage (Diretrizes.txt: training stability is the
explicit top priority). This is deliberately the raw per-step margin, not
a difference/potential-shaped reward and not a sparse +1-on-success signal.

The margin and the success flag (used for `terminated`) are obtained from
`coxeter.spherical_margin_and_test` in a single call: `step()` is called
once per environment per environment-step -- the hottest loop in this
package -- so computing the same eigendecomposition twice here (once via
`spherical_margin`, once via `is_spherical`) would double that cost for no
reason. See `spherical_margin_and_test`'s docstring in coxeter.py.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

# Discover the project root (the directory containing src/) and put it on
# sys.path, exactly as exhaustive.py does, so this file can be imported or
# run directly regardless of the current working directory.
project_root = Path.cwd().resolve()
while not (project_root / "src").exists() and project_root != project_root.parent:
    project_root = project_root.parent
if not (project_root / "src").exists():
    project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import numpy as np

from src.coxeter import (
    LABELS,
    empty_diagram,
    pair_index_list,
    spherical_margin_and_test,
)
from src.datagen import sample_diagram, DATASET_A_P_NONE, DATASET_B_P_NONE

N_MAX = 8  # fixed project-wide maximum (Diretrizes.txt); episodes may use n_active <= N_MAX

_PAIRS: dict[int, list[tuple[int, int]]] = {}


def pairs_for(n: int) -> list[tuple[int, int]]:
    """Cached `pair_index_list(n)` -- the fixed ordering of vertex pairs
    that both `encode_action`/`decode_action` and policy.py's readout head
    rely on for their shared indexing."""
    if n not in _PAIRS:
        _PAIRS[n] = pair_index_list(n)
    return _PAIRS[n]


def num_actions(n: int) -> int:
    """C(n, 2) * len(LABELS): 112 for the project's fixed n = N_MAX = 8.
    This is always evaluated at N_MAX (never at n_active) -- see module
    docstring, "Action space"."""
    return len(pairs_for(n)) * len(LABELS)


def encode_action(pair_idx: int, label: int) -> int:
    """Inverse of `decode_action`. `label` is an actual label value (one of
    LABELS), not its index."""
    label_idx = LABELS.index(label)
    return pair_idx * len(LABELS) + label_idx


def decode_action(action: int, n: int) -> tuple[int, int, int]:
    """action -> (i, j, label): the pair to edit and the label to write."""
    pair_idx, label_idx = divmod(action, len(LABELS))
    i, j = pairs_for(n)[pair_idx]
    return i, j, LABELS[label_idx]


_P_NONE_BY_DATASET = {"A": DATASET_A_P_NONE, "B": DATASET_B_P_NONE}


@dataclass
class StepInfo:
    """Auxiliary info returned alongside every `DynkinEnv.step` call.
    Kept separate from the (state, reward, terminated, truncated) tuple so
    the training loop can ignore it entirely when it doesn't need it."""

    margin: float  # raw (un-squashed) spherical_margin of the new state (== the real submatrix's, see module docstring)
    success: bool  # True iff the episode ended because the diagram is spherical
    steps: int  # number of steps taken so far this episode (after this one)
    n_active: int  # number of real (non-padding) vertices in THIS episode


class DynkinEnv:
    """One episode's worth of state for Objective 1. Deliberately NOT a
    Gymnasium subclass: the interface below (`reset`, `step`) is the whole
    of what train_ppo.py needs, and adding a new dependency for it would
    run against the project's stated preference for the conceptually
    simplest option whenever a choice doesn't affect training stability.
    """

    def __init__(
        self,
        n_max: int = N_MAX,
        active_ns: Optional[Sequence[int]] = None,
        max_steps: int = 30,
        dataset: str = "A",
        reward_squash: str = "tanh",
        seed: Optional[int] = None,
    ):
        if dataset not in _P_NONE_BY_DATASET:
            raise ValueError(f"dataset must be 'A' or 'B', got {dataset!r}")
        if reward_squash not in ("tanh", "clip"):
            raise ValueError(f"reward_squash must be 'tanh' or 'clip', got {reward_squash!r}")
        # active_ns defaults to {n_max} -- i.e. no padding in play at all,
        # which reproduces the old fixed-N=8 behaviour exactly. Curriculum
        # training (train_ppo.py) overrides this via set_active_ns/direct
        # attribute mutation once envs already exist.
        active_ns = (n_max,) if active_ns is None else tuple(active_ns)
        if any(not (2 <= v <= n_max) for v in active_ns):
            raise ValueError(f"active_ns values must lie in [2, n_max={n_max}], got {active_ns}")
        self.n_max = n_max
        self.active_ns = active_ns
        self.max_steps = max_steps
        self.p_none = _P_NONE_BY_DATASET[dataset]
        self.reward_squash = reward_squash
        self.rng = np.random.default_rng(seed)

        self.action_dim = num_actions(n_max)

        self.n_active = active_ns[0]
        self.state: np.ndarray = empty_diagram(n_max)
        self._step_count = 0

    def _squash(self, margin: float) -> float:
        if self.reward_squash == "tanh":
            return float(np.tanh(margin))
        return float(np.clip(margin, -1.0, 1.0))

    def reset(self, seed: Optional[int] = None) -> tuple[np.ndarray, int]:
        """Returns (state, n_active). `n_active` MUST be kept alongside the
        state by the caller (see module docstring) -- it cannot be
        recovered from `state` alone."""
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.n_active = int(self.rng.choice(self.active_ns))
        sub = sample_diagram(self.n_active, self.rng, self.p_none)
        self.state = empty_diagram(self.n_max)
        self.state[: self.n_active, : self.n_active] = sub
        self._step_count = 0
        return self.state.copy(), self.n_active

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, StepInfo]:
        if not (0 <= action < self.action_dim):
            raise ValueError(f"action {action} out of range [0, {self.action_dim})")
        i, j, label = decode_action(action, self.n_max)
        if i >= self.n_active or j >= self.n_active:
            # Should never happen if the caller applied policy.py's action
            # mask correctly -- this is a defensive check, not a normal
            # code path, so it fails loudly rather than silently corrupting
            # a padding vertex's isolation (which the reward's correctness
            # depends on, see module docstring).
            raise ValueError(
                f"action {action} touches padding vertex (pair ({i},{j}), "
                f"n_active={self.n_active}) -- caller must mask invalid actions"
            )
        self.state[i, j] = label
        self.state[j, i] = label
        self._step_count += 1

        # Computed on the FULL (n_max, n_max) state, with no slicing: this
        # is exactly the real submatrix's margin/success, by the isolated-
        # padding argument in the module docstring.
        margin, success = spherical_margin_and_test(self.state)
        reward = self._squash(margin)
        truncated = (not success) and (self._step_count >= self.max_steps)
        terminated = success

        info = StepInfo(margin=margin, success=success, steps=self._step_count, n_active=self.n_active)
        return self.state.copy(), reward, terminated, truncated, info


class VecDynkinEnv:
    """A list of independent `DynkinEnv` instances stepped together.

    Kept deliberately simple (a Python loop, not subprocesses or shared
    memory): at N_MAX = 8 a single step is microseconds of numpy work, so
    the loop overhead is irrelevant next to the cost of the policy's
    forward pass, which already runs on the whole batch at once.
    """

    def __init__(self, n_envs: int, seed: int = 0, **env_kwargs):
        self.n_envs = n_envs
        self.envs = [
            DynkinEnv(seed=seed + k, **env_kwargs) for k in range(n_envs)
        ]

    def set_active_ns(self, active_ns: Sequence[int]) -> None:
        """Update every sub-environment's sampling range for n_active (see
        DynkinEnv.active_ns). Takes effect from each env's NEXT `reset()`
        onward -- an episode already in progress keeps its current
        n_active until it terminates/truncates, so a curriculum transition
        (train_ppo.py) rolls in gradually across the n_envs sub-environments
        rather than discontinuously resetting everything at once."""
        active_ns = tuple(active_ns)
        for env in self.envs:
            env.active_ns = active_ns

    def reset(self) -> tuple[np.ndarray, np.ndarray]:
        """Returns (states, n_actives), both stacked over n_envs."""
        results = [env.reset() for env in self.envs]
        states = np.stack([s for s, _ in results])
        n_actives = np.array([n for _, n in results], dtype=np.int64)
        return states, n_actives

    def step(
        self, actions: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[StepInfo]]:
        """`actions` has shape (n_envs,). Any environment that terminates or
        truncates is reset immediately (standard vectorised-env
        auto-reset), and the returned `states`/`n_actives` already describe
        the FRESH post-reset episode where that happened -- `infos[k]`
        still describes the episode that just ended (its OWN n_active
        included), which is what the training loop needs for logging
        (final margin, success, and which N that success/failure was on).
        """
        n_max = self.envs[0].n_max
        states = np.empty((self.n_envs, n_max, n_max), dtype=np.int64)
        rewards = np.empty(self.n_envs, dtype=np.float64)
        dones = np.empty(self.n_envs, dtype=bool)
        n_actives = np.empty(self.n_envs, dtype=np.int64)
        infos: list[StepInfo] = []
        for k, env in enumerate(self.envs):
            s, r, terminated, truncated, info = env.step(int(actions[k]))
            done = terminated or truncated
            if done:
                s, _ = env.reset()
            states[k] = s
            rewards[k] = r
            dones[k] = done
            n_actives[k] = env.n_active  # the FRESH episode's n_active if done, else unchanged
            infos.append(info)
        return states, rewards, dones, n_actives, infos


# --------------------------------------------------------------------------- #
# Sanity checks -- run this file directly. Missing until now; added to match
# the project-wide convention (docstring + sanity checks in __main__) that
# coxeter.py, exhaustive.py and datagen.py already follow.
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    from src.coxeter import is_spherical, diagram_from_edges, spherical_margin

    n = N_MAX

    # encode_action / decode_action must be mutual inverses over the whole
    # action space, and num_actions must match C(n,2) * len(LABELS).
    assert num_actions(n) == 28 * 4 == 112
    for a in range(num_actions(n)):
        i, j, lab = decode_action(a, n)
        pair_idx = pairs_for(n).index((i, j))
        assert encode_action(pair_idx, lab) == a
    print("encode/decode_action: round-trip OK over all", num_actions(n), "actions")

    # reset() must produce a valid, symmetric labelled diagram in
    # coxeter.py's own representation, at full N_MAX (default active_ns).
    env = DynkinEnv(n_max=n, max_steps=10, dataset="A", seed=0)
    s0, n_active0 = env.reset()
    assert n_active0 == n  # default active_ns == (n_max,)
    assert s0.shape == (n, n)
    assert np.array_equal(s0, s0.T)
    assert np.all(np.diag(s0) == 0)
    off_diag = s0[~np.eye(n, dtype=bool)]
    assert set(np.unique(off_diag)) <= set(LABELS)
    print("reset(): valid symmetric diagram OK (active_ns defaults to full n_max)")

    # step() must write exactly the requested edge, and success/terminated
    # must agree with is_spherical on the resulting state.
    env.state = empty_diagram(n)
    env.n_active = n
    before = env.state.copy()
    a = encode_action(pair_idx=0, label=3)  # pair 0 == (0, 1) in pair_index_list(n)
    new_state, reward, terminated, truncated, info = env.step(a)
    diff_mask = new_state != before
    assert diff_mask.sum() == 2 and diff_mask[0, 1] and diff_mask[1, 0]
    assert new_state[0, 1] == 3 and new_state[1, 0] == 3
    assert info.success == is_spherical(new_state) == terminated
    assert info.n_active == n
    print("step(): writes exactly the requested edge; success/terminated consistent")

    # A no-op edit on an already-spherical diagram must terminate successfully.
    A8 = diagram_from_edges(n, [(k, k + 1, 3) for k in range(n - 1)])  # A_8, a path
    assert is_spherical(A8)
    env.state = A8.copy()
    a_noop = encode_action(pair_idx=0, label=3)  # rewrite (0,1) with its own label
    _, reward, terminated, truncated, info = env.step(a_noop)
    assert terminated and info.success and not truncated
    print("step(): no-op edit on an already-spherical diagram terminates successfully")

    # Reward must stay inside [-1, 1] whatever the squash mode.
    for squash in ("tanh", "clip"):
        env2 = DynkinEnv(n_max=n, reward_squash=squash, seed=1)
        dense = diagram_from_edges(n, [(i, j, 6) for i in range(n) for j in range(i + 1, n)])
        env2.state = dense
        env2.n_active = n
        _, r, *_ = env2.step(encode_action(0, 6))
        assert -1.0 <= r <= 1.0
    print("reward squashing: stays within [-1, 1] under both 'tanh' and 'clip'")

    # --- Variable N / padding ------------------------------------------------

    # reset() with active_ns < n_max must sample n_active from that set, pad
    # the rest with isolated vertices, and never touch them.
    env3 = DynkinEnv(n_max=8, active_ns=(3, 4, 5), max_steps=10, dataset="A", seed=2)
    seen_ns = set()
    for _ in range(50):
        s, n_active = env3.reset()
        seen_ns.add(n_active)
        assert 3 <= n_active <= 5
        # Off-diagonal padding entries are NO_EDGE (2); the diagonal is always
        # 0 (coxeter.py's "unused" placeholder), padding or not, so exclude it.
        off_diag = ~np.eye(8, dtype=bool)
        touches_padding = np.zeros((8, 8), dtype=bool)
        touches_padding[n_active:, :] = True
        touches_padding[:, n_active:] = True
        assert np.all(s[touches_padding & off_diag] == 2)  # NO_EDGE == 2
    assert seen_ns == {3, 4, 5}, f"expected to see every value in active_ns, got {seen_ns}"
    print("reset(): active_ns={3,4,5} -> n_active sampled from that set; padding stays isolated")

    # An action touching a padding vertex must be rejected (defensive check
    # -- the caller, i.e. policy.py's action mask, should never produce one).
    env3.state, env3.n_active = empty_diagram(8), 3
    pair_idx_padding = pairs_for(8).index((3, 4))  # both indices are padding when n_active=3
    bad_action = encode_action(pair_idx_padding, 3)
    try:
        env3.step(bad_action)
        raise AssertionError("expected a ValueError for an action touching a padding vertex")
    except ValueError:
        pass
    print("step(): action touching a padding vertex is rejected")

    # The core mathematical fact this whole design leans on: padding with
    # isolated vertices must not change spherical_margin/is_spherical
    # relative to the real (n_active, n_active) submatrix, for both a
    # spherical and a non-spherical example.
    D4 = diagram_from_edges(4, [(0, 1, 3), (0, 2, 3), (0, 3, 3)])  # spherical
    padded_D4 = empty_diagram(8)
    padded_D4[:4, :4] = D4
    assert is_spherical(padded_D4) == is_spherical(D4) == True
    assert abs(spherical_margin(padded_D4) - spherical_margin(D4)) < 1e-9

    dense_cycle = diagram_from_edges(3, [(0, 1, 3), (1, 2, 4), (0, 2, 6)])  # not spherical
    padded_dense = empty_diagram(8)
    padded_dense[:3, :3] = dense_cycle
    assert is_spherical(padded_dense) == is_spherical(dense_cycle) == False
    assert abs(spherical_margin(padded_dense) - spherical_margin(dense_cycle)) < 1e-9
    print("padding-with-isolated-vertices: spherical_margin/is_spherical exactly match the real submatrix")

    # An episode run entirely at n_active < n_max must reach the SAME
    # success verdict via step() (full n_max state) as testing the sliced
    # submatrix directly.
    env4 = DynkinEnv(n_max=8, active_ns=(4,), max_steps=1, dataset="A", seed=3)
    env4.state = empty_diagram(8)
    env4.n_active = 4
    # Build A_4 (path, all labels 3) by three edits, one at a time, matching
    # pair_index_list(4) = (0,1),(0,2),(0,3),(1,2),(1,3),(2,3) restricted to n=4
    # inside the n_max=8 pair ordering.
    for (u, v) in [(0, 1), (1, 2), (2, 3)]:
        pidx = pairs_for(8).index((u, v))
        env4.step(encode_action(pidx, 3))
    assert is_spherical(env4.state[:4, :4])
    off_diag8 = ~np.eye(8, dtype=bool)
    touches_padding8 = np.zeros((8, 8), dtype=bool)
    touches_padding8[4:, :] = True
    touches_padding8[:, 4:] = True
    assert np.all(env4.state[touches_padding8 & off_diag8] == 2)
    print("step(): building A_4 inside an n_max=8 padded state keeps padding isolated throughout")

    # VecDynkinEnv: shapes (states, n_actives), and auto-reset must not crash
    # across many steps (max_steps=3 guarantees every env truncates at least
    # once in 10 steps). Uses active_ns=(3,4,5,6,7,8) to also exercise padding.
    vec = VecDynkinEnv(n_envs=4, seed=0, n_max=n, active_ns=(3, 4, 5, 6, 7, 8), max_steps=3, dataset="B")
    states, n_actives = vec.reset()
    assert states.shape == (4, n, n) and n_actives.shape == (4,)
    assert np.all((n_actives >= 3) & (n_actives <= 8))
    rng_test = np.random.default_rng(0)
    for _ in range(10):
        actions = rng_test.integers(0, vec.envs[0].action_dim, size=4)
        # Respect the action mask by hand here (a real caller uses policy.py's
        # mask): clamp each sampled action down to a pair inside n_active.
        safe_actions = np.empty(4, dtype=np.int64)
        for k in range(4):
            n_act = n_actives[k]
            valid_pairs = [t for t, (i, j) in enumerate(pairs_for(n)) if i < n_act and j < n_act]
            pidx = valid_pairs[actions[k] % len(valid_pairs)]
            label_idx = actions[k] % len(LABELS)
            safe_actions[k] = pidx * len(LABELS) + label_idx
        states, rewards, dones, n_actives, infos = vec.step(safe_actions)
        assert states.shape == (4, n, n)
        assert rewards.shape == (4,) and dones.shape == (4,) and n_actives.shape == (4,)
        assert len(infos) == 4
    print("VecDynkinEnv: shapes OK across steps (incl. n_actives), auto-reset does not crash")

    # set_active_ns must change what future resets sample, without touching
    # in-flight episodes.
    vec.set_active_ns((3,))
    for env_ in vec.envs:
        assert env_.active_ns == (3,)
    print("VecDynkinEnv.set_active_ns: updates every sub-env's sampling range")

    print("env.py: all sanity checks passed.")

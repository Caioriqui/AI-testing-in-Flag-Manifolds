"""
env.py -- RL environment for Objective 1 (MM845): building spherical
Coxeter diagrams by single-edge edits.

State
-----
The state is a labelled diagram on N = 8 vertices, in exactly the
representation used by coxeter.py: a symmetric (N, N) int64 array with
values in LABELS = (2, 3, 4, 6) off the diagonal (2 == "no edge") and 0 on
the diagonal. N is fixed at 8 project-wide (Diretrizes.txt); nothing below
is written to generalise to other N by design.

Action space
------------
A single discrete action combines "which unordered vertex pair" and "which
label to write there" (Diretrizes.txt explicitly rejects factoring this
into two sequential decisions as unneeded complexity at this scale). There
are C(8, 2) = 28 pairs and 4 labels, so `NUM_ACTIONS = 28 * 4 = 112`. The
encoding

    action = pair_index * len(LABELS) + label_index

is the single source of truth for this mapping (`encode_action` /
`decode_action` below); policy.py imports it from here rather than
re-deriving it, so the network's output ordering and the environment's
action semantics can never drift apart.

Episode
-------
Each episode starts from a diagram sampled i.i.d. by datagen.py (Dataset A
or B -- configurable), never from a corrupted known-valid target, so the
agent never gets to reverse-engineer a target it was implicitly shown
(Diretrizes.txt, and the project's stated data-generation rationale). At
every step the agent overwrites one entry M[i, j] = M[j, i] with the label
its action names -- including a "no-op" of writing back the label already
there. An episode ends when the diagram becomes spherical (success) or a
fixed step budget is exhausted (truncation).

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
from typing import Optional

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

N_DEFAULT = 8  # fixed project-wide (Diretrizes.txt)

_PAIRS: dict[int, list[tuple[int, int]]] = {}


def pairs_for(n: int) -> list[tuple[int, int]]:
    """Cached `pair_index_list(n)` -- the fixed ordering of vertex pairs
    that both `encode_action`/`decode_action` and policy.py's readout head
    rely on for their shared indexing."""
    if n not in _PAIRS:
        _PAIRS[n] = pair_index_list(n)
    return _PAIRS[n]


def num_actions(n: int) -> int:
    """C(n, 2) * len(LABELS): 112 for the project's fixed n = 8."""
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

    margin: float  # raw (un-squashed) spherical_margin of the new state
    success: bool  # True iff the episode ended because the diagram is spherical
    steps: int  # number of steps taken so far this episode (after this one)


class DynkinEnv:
    """One episode's worth of state for Objective 1. Deliberately NOT a
    Gymnasium subclass: the interface below (`reset`, `step`) is the whole
    of what train_ppo.py needs, and adding a new dependency for it would
    run against the project's stated preference for the conceptually
    simplest option whenever a choice doesn't affect training stability.
    """

    def __init__(
        self,
        n: int = N_DEFAULT,
        max_steps: int = 30,
        dataset: str = "A",
        reward_squash: str = "tanh",
        seed: Optional[int] = None,
    ):
        if dataset not in _P_NONE_BY_DATASET:
            raise ValueError(f"dataset must be 'A' or 'B', got {dataset!r}")
        if reward_squash not in ("tanh", "clip"):
            raise ValueError(f"reward_squash must be 'tanh' or 'clip', got {reward_squash!r}")
        self.n = n
        self.max_steps = max_steps
        self.p_none = _P_NONE_BY_DATASET[dataset]
        self.reward_squash = reward_squash
        self.rng = np.random.default_rng(seed)

        self._pairs = pairs_for(n)
        self.action_dim = num_actions(n)

        self.state: np.ndarray = empty_diagram(n)
        self._step_count = 0

    def _squash(self, margin: float) -> float:
        if self.reward_squash == "tanh":
            return float(np.tanh(margin))
        return float(np.clip(margin, -1.0, 1.0))

    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.state = sample_diagram(self.n, self.rng, self.p_none)
        self._step_count = 0
        return self.state.copy()

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, StepInfo]:
        if not (0 <= action < self.action_dim):
            raise ValueError(f"action {action} out of range [0, {self.action_dim})")
        i, j, label = decode_action(action, self.n)
        self.state[i, j] = label
        self.state[j, i] = label
        self._step_count += 1

        margin, success = spherical_margin_and_test(self.state)
        reward = self._squash(margin)
        truncated = (not success) and (self._step_count >= self.max_steps)
        terminated = success

        info = StepInfo(margin=margin, success=success, steps=self._step_count)
        return self.state.copy(), reward, terminated, truncated, info


class VecDynkinEnv:
    """A list of independent `DynkinEnv` instances stepped together.

    Kept deliberately simple (a Python loop, not subprocesses or shared
    memory): at N = 8 a single step is microseconds of numpy work, so the
    loop overhead is irrelevant next to the cost of the policy's forward
    pass, which already runs on the whole batch at once.
    """

    def __init__(self, n_envs: int, seed: int = 0, **env_kwargs):
        self.n_envs = n_envs
        self.envs = [
            DynkinEnv(seed=seed + k, **env_kwargs) for k in range(n_envs)
        ]

    def reset(self) -> np.ndarray:
        states = [env.reset() for env in self.envs]
        return np.stack(states)

    def step(
        self, actions: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[StepInfo]]:
        """`actions` has shape (n_envs,). Any environment that terminates or
        truncates is reset immediately (standard vectorised-env
        auto-reset), and the returned state is already the FRESH state --
        `infos[k]` still describes the episode that just ended, which is
        what the training loop needs for logging (final margin, success)."""
        states = np.empty((self.n_envs, self.envs[0].n, self.envs[0].n), dtype=np.int64)
        rewards = np.empty(self.n_envs, dtype=np.float64)
        dones = np.empty(self.n_envs, dtype=bool)
        infos: list[StepInfo] = []
        for k, env in enumerate(self.envs):
            s, r, terminated, truncated, info = env.step(int(actions[k]))
            done = terminated or truncated
            if done:
                s = env.reset()
            states[k] = s
            rewards[k] = r
            dones[k] = done
            infos.append(info)
        return states, rewards, dones, infos


# --------------------------------------------------------------------------- #
# Sanity checks -- run this file directly. Missing until now; added to match
# the project-wide convention (docstring + sanity checks in __main__) that
# coxeter.py, exhaustive.py and datagen.py already follow.
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    from src.coxeter import is_spherical, diagram_from_edges

    n = N_DEFAULT

    # encode_action / decode_action must be mutual inverses over the whole
    # action space, and num_actions must match C(n,2) * len(LABELS).
    assert num_actions(n) == 28 * 4 == 112
    for a in range(num_actions(n)):
        i, j, lab = decode_action(a, n)
        pair_idx = pairs_for(n).index((i, j))
        assert encode_action(pair_idx, lab) == a
    print("encode/decode_action: round-trip OK over all", num_actions(n), "actions")

    # reset() must produce a valid, symmetric labelled diagram in
    # coxeter.py's own representation.
    env = DynkinEnv(n=n, max_steps=10, dataset="A", seed=0)
    s0 = env.reset()
    assert s0.shape == (n, n)
    assert np.array_equal(s0, s0.T)
    assert np.all(np.diag(s0) == 0)
    off_diag = s0[~np.eye(n, dtype=bool)]
    assert set(np.unique(off_diag)) <= set(LABELS)
    print("reset(): valid symmetric diagram OK")

    # step() must write exactly the requested edge, and success/terminated
    # must agree with is_spherical on the resulting state.
    env.state = empty_diagram(n)
    before = env.state.copy()
    a = encode_action(pair_idx=0, label=3)  # pair 0 == (0, 1) in pair_index_list(n)
    new_state, reward, terminated, truncated, info = env.step(a)
    diff_mask = new_state != before
    assert diff_mask.sum() == 2 and diff_mask[0, 1] and diff_mask[1, 0]
    assert new_state[0, 1] == 3 and new_state[1, 0] == 3
    assert info.success == is_spherical(new_state) == terminated
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
        env2 = DynkinEnv(n=n, reward_squash=squash, seed=1)
        dense = diagram_from_edges(n, [(i, j, 6) for i in range(n) for j in range(i + 1, n)])
        env2.state = dense
        _, r, *_ = env2.step(encode_action(0, 6))
        assert -1.0 <= r <= 1.0
    print("reward squashing: stays within [-1, 1] under both 'tanh' and 'clip'")

    # VecDynkinEnv: shapes, and auto-reset must not crash across many steps
    # (max_steps=3 guarantees every env truncates at least once in 10 steps).
    vec = VecDynkinEnv(n_envs=4, seed=0, n=n, max_steps=3, dataset="B")
    states = vec.reset()
    assert states.shape == (4, n, n)
    rng_test = np.random.default_rng(0)
    for _ in range(10):
        actions = rng_test.integers(0, vec.envs[0].action_dim, size=4)
        states, rewards, dones, infos = vec.step(actions)
        assert states.shape == (4, n, n)
        assert rewards.shape == (4,) and dones.shape == (4,)
        assert len(infos) == 4
    print("VecDynkinEnv: shapes OK across steps, auto-reset does not crash")

    print("env.py: all sanity checks passed.")

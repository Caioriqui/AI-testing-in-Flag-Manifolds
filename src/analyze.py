"""
analyze.py -- load a trained PPO checkpoint (train_ppo.py) and report on
what it learned: rollouts on fresh initial diagrams, a best-effort
classification of the spherical diagrams it reaches by Coxeter type, and
the training curves logged during training. This module is what
notebooks/02_train_rl.ipynb calls into (Diretrizes.txt): all the substance
lives here, in src/, and the notebook is only a thin, re-runnable
demonstration on top of it.

Type classification
--------------------
`classify_spherical_diagram` uses the standard fact that a *connected*
spherical diagram is a tree (Coxeter's classification theorem). That is
the right tool here, because this module's job is to describe and
visualise diagrams already independently confirmed spherical by
`is_spherical` -- it never uses the tree fact to decide sphericity itself.
Contrast exhaustive.py, whose entire point is to never assume that fact
because it is what is being cross-checked there.

The classifier covers exactly the types reachable with LABELS = (2,3,4,6)
(coxeter.py's project-wide convention): A_n, B_n/C_n, F_4, G_2 (the
non-simply-laced families), and D_n, E_6, E_7, E_8 (the simply-laced
branching families). H_3, H_4 and I_2(m>=5) are unreachable by
construction (they need a label >= 5) and never appear.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

project_root = Path.cwd().resolve()
while not (project_root / "src").exists() and project_root != project_root.parent:
    project_root = project_root.parent
if not (project_root / "src").exists():
    project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import numpy as np
import torch
from torch.distributions import Categorical

from src.coxeter import NO_EDGE, UnionFind, diagram_from_edges, edge_list, is_spherical
from src.env import N_DEFAULT, DynkinEnv
from src.policy import DynkinGNN


# --------------------------------------------------------------------------- #
# Checkpoint loading and rollouts
# --------------------------------------------------------------------------- #

def load_checkpoint(path, device: str = "cpu") -> tuple[DynkinGNN, dict]:
    """Load a checkpoint saved by train_ppo.save_checkpoint. Returns the
    model (in eval mode) and the config dict it was trained with."""
    ckpt = torch.load(path, map_location=device)
    cfg = ckpt["config"]
    model = DynkinGNN(
        n=cfg["n"],
        hidden_dim=cfg["hidden_dim"],
        label_embed_dim=cfg["label_embed_dim"],
        n_layers=cfg["n_layers"],
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, cfg


def rollout_episode(model: DynkinGNN, env: DynkinEnv, deterministic: bool = False) -> dict:
    """Run one episode with `model` acting in `env`. `deterministic=True`
    takes the argmax action at every step (useful for a clean demo);
    `deterministic=False` samples, matching what the agent actually saw
    during training."""
    state = env.reset()
    states = [state.copy()]
    margins: list[float] = []
    actions: list[int] = []
    success = False

    with torch.no_grad():
        for _ in range(env.max_steps):
            logits, _value = model(state[None, ...])
            if deterministic:
                action = int(torch.argmax(logits, dim=-1).item())
            else:
                action = int(Categorical(logits=logits).sample().item())

            state, _reward, terminated, truncated, info = env.step(action)
            states.append(state.copy())
            margins.append(info.margin)
            actions.append(action)

            if terminated or truncated:
                success = info.success
                break

    return dict(states=states, margins=margins, actions=actions, success=success, n_steps=len(actions))


def evaluate(
    model: DynkinGNN,
    n_episodes: int = 200,
    dataset: str = "A",
    n: int = N_DEFAULT,
    max_episode_steps: int = 30,
    seed: int = 0,
    deterministic: bool = False,
) -> dict:
    """Roll out `n_episodes` fresh episodes and summarise success."""
    env = DynkinEnv(n=n, max_steps=max_episode_steps, dataset=dataset, seed=seed)
    n_success = 0
    steps_to_success: list[int] = []
    final_diagrams: list[np.ndarray] = []

    for _ in range(n_episodes):
        traj = rollout_episode(model, env, deterministic=deterministic)
        if traj["success"]:
            n_success += 1
            steps_to_success.append(traj["n_steps"])
            final_diagrams.append(traj["states"][-1])

    return dict(
        n_episodes=n_episodes,
        success_rate=n_success / n_episodes,
        mean_steps_to_success=float(np.mean(steps_to_success)) if steps_to_success else float("nan"),
        final_diagrams=final_diagrams,
    )


# --------------------------------------------------------------------------- #
# Coxeter-type classification of (already confirmed) spherical diagrams
# --------------------------------------------------------------------------- #

def _component_vertex_sets(M: np.ndarray) -> list[list[int]]:
    """Vertex sets of each connected component of M's underlying graph."""
    n = M.shape[0]
    uf = UnionFind(n)
    for i, j in edge_list(M):
        uf.union(i, j)
    comps: dict[int, list[int]] = {}
    for v in range(n):
        comps.setdefault(uf.find(v), []).append(v)
    return list(comps.values())


def _classify_component(M: np.ndarray, vertices: list[int]) -> str:
    """Classify one connected component, ASSUMING the full diagram M is
    spherical (so, by Coxeter's theorem, this component's underlying graph
    is a tree -- used here only for identification, see module docstring).
    """
    k = len(vertices)
    if k == 1:
        return "A_1"

    sub = M[np.ix_(vertices, vertices)]
    # Explicit adjacency, with the diagonal excluded once and for all: the
    # diagonal holds 0 (coxeter.py's "unused" placeholder), which is not a
    # real label and is NOT equal to NO_EDGE (2) -- every neighbour lookup
    # below goes through `adj`, never a raw `sub[...] != NO_EDGE` check, so
    # a diagonal entry can never be mistaken for a self-edge again.
    adj = (sub != NO_EDGE) & ~np.eye(k, dtype=bool)
    deg = adj.sum(axis=1).tolist()
    max_deg = max(deg)

    if max_deg <= 2:
        # A path. Walk it from an endpoint to read off the edge labels in order.
        endpoints = [t for t in range(k) if deg[t] <= 1]
        start = endpoints[0] if endpoints else 0
        order = [start]
        visited = {start}
        cur = start
        while len(order) < k:
            nxt = [t for t in range(k) if adj[cur, t] and t not in visited]
            if not nxt:
                break
            cur = nxt[0]
            order.append(cur)
            visited.add(cur)
        edge_labels = [int(sub[order[t], order[t + 1]]) for t in range(k - 1)]
        distinct = set(edge_labels)

        if distinct == {3}:
            return f"A_{k}"
        if 6 in distinct:
            if k == 2:
                return "G_2"
            return f"unknown (path with a 6-label edge, k={k}, labels={edge_labels})"
        if 4 in distinct:
            count4 = edge_labels.count(4)
            if count4 == 1:
                pos = edge_labels.index(4)
                if pos in (0, k - 2):
                    return f"B_{k}/C_{k}"
                if k == 4 and pos == 1:
                    return "F_4"
            return f"unknown (path with label(s) 4, k={k}, labels={edge_labels})"
        return f"unknown (path, labels={edge_labels})"

    # A branching tree. Every classical branching type (D_n, E_6, E_7, E_8)
    # is simply-laced (labels all 3) with exactly one degree-3 vertex.
    branch_candidates = [t for t in range(k) if deg[t] == 3]
    if max_deg > 3 or len(branch_candidates) != 1:
        return f"unknown (non-classical branching, degrees={deg})"

    labels_present = {int(sub[i, j]) for i in range(k) for j in range(i + 1, k) if sub[i, j] != NO_EDGE}
    if labels_present - {3}:
        return f"unknown (branched diagram with a non-3 label, labels={labels_present})"

    branch = branch_candidates[0]
    arm_lengths = []
    for nb in (t for t in range(k) if sub[branch, t] != NO_EDGE):
        length, prev, cur = 1, branch, nb
        while True:
            nxt = [t for t in range(k) if sub[cur, t] != NO_EDGE and t != prev]
            if not nxt:
                break
            prev, cur = cur, nxt[0]
            length += 1
        arm_lengths.append(length)
    arm_lengths.sort()

    if arm_lengths == [1, 1, k - 3]:
        return f"D_{k}"
    if k == 6 and arm_lengths == [1, 2, 2]:
        return "E_6"
    if k == 7 and arm_lengths == [1, 2, 3]:
        return "E_7"
    if k == 8 and arm_lengths == [1, 2, 4]:
        return "E_8"
    return f"unknown (branched, arm_lengths={arm_lengths}, k={k})"


def classify_spherical_diagram(M: np.ndarray) -> list[str]:
    """One Coxeter-type label per connected component of M. Raises if M is
    not spherical -- this function describes spherical diagrams, it does
    not test for sphericity (use `is_spherical` for that)."""
    if not is_spherical(M):
        raise ValueError("classify_spherical_diagram expects a spherical diagram")
    return [_classify_component(M, comp) for comp in _component_vertex_sets(M)]


# --------------------------------------------------------------------------- #
# Training curves
# --------------------------------------------------------------------------- #

def plot_training_curves(log_path, out_path: Optional[str] = None):
    import matplotlib.pyplot as plt

    data = np.load(log_path)
    fig, axes = plt.subplots(3, 1, figsize=(7, 9), sharex=True)
    axes[0].plot(data["timesteps"], data["reward"])
    axes[0].set_ylabel("mean reward / step")
    axes[1].plot(data["timesteps"], data["margin"])
    axes[1].set_ylabel("mean spherical_margin")
    axes[2].plot(data["timesteps"], data["success_rate"])
    axes[2].set_ylabel("success rate")
    axes[2].set_xlabel("environment steps")
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=150)
        print(f"Saved training curves to {out_path}")
    return fig


# --------------------------------------------------------------------------- #
# Self-test of the classifier against known examples (independent of any
# trained checkpoint) -- run with `python -m src.analyze --self-test`
# --------------------------------------------------------------------------- #

def _self_test_classification() -> None:
    A3 = diagram_from_edges(4, [(0, 1, 3), (1, 2, 3), (2, 3, 3)])
    assert classify_spherical_diagram(A3) == ["A_4"]

    D5 = diagram_from_edges(5, [(0, 1, 3), (1, 2, 3), (1, 3, 3), (3, 4, 3)])
    assert classify_spherical_diagram(D5) == ["D_5"], classify_spherical_diagram(D5)

    E6 = diagram_from_edges(6, [(0, 1, 3), (1, 2, 3), (2, 3, 3), (3, 4, 3), (2, 5, 3)])
    assert classify_spherical_diagram(E6) == ["E_6"], classify_spherical_diagram(E6)

    B4 = diagram_from_edges(4, [(0, 1, 3), (1, 2, 3), (2, 3, 4)])
    assert classify_spherical_diagram(B4) == ["B_4/C_4"], classify_spherical_diagram(B4)

    F4 = diagram_from_edges(4, [(0, 1, 3), (1, 2, 4), (2, 3, 3)])
    assert classify_spherical_diagram(F4) == ["F_4"], classify_spherical_diagram(F4)

    G2 = diagram_from_edges(2, [(0, 1, 6)])
    assert classify_spherical_diagram(G2) == ["G_2"]

    # A disjoint union: A_2 x A_1 x A_1 on 4 vertices.
    disjoint = diagram_from_edges(4, [(0, 1, 3)])
    labels = sorted(classify_spherical_diagram(disjoint))
    assert labels == ["A_1", "A_1", "A_2"], labels

    print("analyze.py: classifier self-test passed (A_4, D_5, E_6, B_4/C_4, F_4, G_2, A_2+A_1+A_1).")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate and analyze a trained Objective-1 PPO checkpoint.")
    p.add_argument("--checkpoint", type=str, default="results/checkpoint_latest.pt")
    p.add_argument("--n-episodes", type=int, default=200)
    p.add_argument("--dataset", type=str, default="A", choices=["A", "B"])
    p.add_argument("--deterministic", action="store_true")
    p.add_argument("--plot", type=str, default=None, help="output path for the training-curve figure")
    p.add_argument("--self-test", action="store_true", help="run the classifier self-test and exit")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    if args.self_test:
        _self_test_classification()
        sys.exit(0)

    model, cfg = load_checkpoint(args.checkpoint)
    print(f"Loaded checkpoint: {args.checkpoint}")

    results = evaluate(
        model,
        n_episodes=args.n_episodes,
        dataset=args.dataset,
        n=cfg["n"],
        max_episode_steps=cfg["max_episode_steps"],
        deterministic=args.deterministic,
    )
    print(f"Success rate over {results['n_episodes']} episodes: {results['success_rate']:.3f}")
    print(f"Mean steps to success (successful episodes only): {results['mean_steps_to_success']:.2f}")

    print("\nCoxeter type of a few successful final diagrams:")
    for M in results["final_diagrams"][:10]:
        print(" ", " x ".join(classify_spherical_diagram(M)))

    log_path = Path(args.checkpoint).parent / "train_log.npz"
    if log_path.exists():
        out_path = args.plot or str(Path(args.checkpoint).parent / "training_curves.png")
        plot_training_curves(log_path, out_path)
    else:
        print(f"No training log found at {log_path}; skipping plot.")

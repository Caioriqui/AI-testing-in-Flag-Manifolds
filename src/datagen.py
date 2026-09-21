"""
datagen.py -- i.i.d. samplers for initial diagrams s0 (Dataset A / B), and
the pre-training structural sanity checks of report section 2.2.

Each of the C(n, 2) edge labels is drawn i.i.d. from {none, 3, 4, 6}. The
two datasets differ only in P(none):

    Dataset A (uniform):  P(none) = 1/4  -> edge present w.p. 3/4.
                           At N=8, E[#edges] = 3/4 * 28 = 21.
    Dataset B (sparse):   P(none) = 3/4  -> edge present w.p. 1/4.
                           At N=8, E[#edges] = 1/4 * 28 = 7 = N-1.

Dataset B is centred on the tree regime (N-1 edges); Dataset A is centred
far above it, so that almost every sample contains a cycle and is therefore
maximally far from spherical regardless of labelling. Neither construction
ever references a known valid diagram (contrast with "corrupt a target"),
which is the property the report asks for in section 2.1.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.coxeter import (
    LABELS,
    NO_EDGE,
    connected_components,
    empty_diagram,
    is_acyclic,
    is_spherical,
    pair_index_list,
)

DATASET_A_P_NONE = 0.25  # uniform over the 4 labels
DATASET_B_P_NONE = 0.75  # elevated P(none) -> expected N-1 edges


# --------------------------------------------------------------------------- #
# Sampling
# --------------------------------------------------------------------------- #

def _label_probs(p_none: float) -> np.ndarray:
    """Probability vector over LABELS = (2, 3, 4, 6), i.e. (none, 3, 4, 6)."""
    p_edge = (1.0 - p_none) / 3.0
    return np.array([p_none, p_edge, p_edge, p_edge])


def sample_diagram(n: int, rng: np.random.Generator, p_none: float) -> np.ndarray:
    """Sample one labelled diagram on n vertices under the i.i.d. regime."""
    pairs = pair_index_list(n)
    probs = _label_probs(p_none)
    choices = rng.choice(LABELS, size=len(pairs), p=probs)
    M = empty_diagram(n)
    for (i, j), lab in zip(pairs, choices):
        M[i, j] = M[j, i] = lab
    return M


def sample_dataset(n: int, n_samples: int, p_none: float, seed: int) -> np.ndarray:
    """Array of shape (n_samples, n, n) of diagrams sampled i.i.d."""
    rng = np.random.default_rng(seed)
    return np.stack([sample_diagram(n, rng, p_none) for _ in range(n_samples)])


def sample_dataset_A(n: int, n_samples: int, seed: int = 0) -> np.ndarray:
    return sample_dataset(n, n_samples, DATASET_A_P_NONE, seed)


def sample_dataset_B(n: int, n_samples: int, seed: int = 0) -> np.ndarray:
    return sample_dataset(n, n_samples, DATASET_B_P_NONE, seed)


# --------------------------------------------------------------------------- #
# Pre-training structural statistics (report section 2.2)
# --------------------------------------------------------------------------- #

@dataclass
class DatasetStats:
    n: int
    n_samples: int
    p_none: float
    mean_components: float
    std_components: float
    frac_acyclic: float
    frac_spherical: float
    # theoretical expectation of the number of edges under this i.i.d. regime,
    # for comparison against the sample as a sanity check on the generator.
    expected_num_edges: float
    observed_mean_num_edges: float


def compute_stats(diagrams: np.ndarray, p_none: float) -> DatasetStats:
    """Structural statistics of a sampled dataset, matched against their
    theoretical values under the i.i.d. regime (independent of the agent).
    """
    n = diagrams.shape[1]
    n_samples = diagrams.shape[0]
    n_pairs = n * (n - 1) // 2

    comps = np.array([connected_components(M) for M in diagrams])
    acyclic = np.array([is_acyclic(M) for M in diagrams])
    spherical = np.array([is_spherical(M) for M in diagrams])
    n_edges = np.array([int(np.sum(M[np.triu_indices(n, k=1)] != NO_EDGE)) for M in diagrams])

    return DatasetStats(
        n=n,
        n_samples=n_samples,
        p_none=p_none,
        mean_components=float(comps.mean()),
        std_components=float(comps.std()),
        frac_acyclic=float(acyclic.mean()),
        frac_spherical=float(spherical.mean()),
        expected_num_edges=(1.0 - p_none) * n_pairs,
        observed_mean_num_edges=float(n_edges.mean()),
    )


def print_stats(name: str, stats: DatasetStats) -> None:
    print(
        f"Dataset {name} (N={stats.n}, {stats.n_samples} samples, P(none)={stats.p_none}):\n"
        f"  E[#edges] theoretical / observed : {stats.expected_num_edges:.2f} / {stats.observed_mean_num_edges:.2f}\n"
        f"  mean #connected components       : {stats.mean_components:.3f} (std {stats.std_components:.3f})\n"
        f"  P(acyclic)                       : {stats.frac_acyclic:.4f}\n"
        f"  P(already spherical)             : {stats.frac_spherical:.6f}"
    )


if __name__ == "__main__":
    N = 8
    N_SAMPLES = 20_000
    SEED = 0

    diagrams_A = sample_dataset_A(N, N_SAMPLES, seed=SEED)
    diagrams_B = sample_dataset_B(N, N_SAMPLES, seed=SEED)

    stats_A = compute_stats(diagrams_A, DATASET_A_P_NONE)
    stats_B = compute_stats(diagrams_B, DATASET_B_P_NONE)

    print_stats("A (uniform)", stats_A)
    print()
    print_stats("B (sparse)", stats_B)

    # Sanity checks on the generator itself, independent of any agent:
    # Dataset A should sit far above the tree regime (n-1 = 7 edges at N=8)
    # and be acyclic only in a vanishing fraction of samples; Dataset B
    # should sit right on it and be acyclic much more often.
    assert stats_A.expected_num_edges > N - 1
    assert stats_B.expected_num_edges == N - 1
    assert stats_A.frac_acyclic < stats_B.frac_acyclic
    print("\ndatagen.py: all sanity checks passed.")

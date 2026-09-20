"""
exhaustive.py -- exhaustive test of the spherical condition, up to N = 6.

This tests EVERY labelled diagram on N vertices -- including every diagram
whose underlying graph has a cycle -- against `is_spherical`. It does not
rely on the fact that a connected spherical diagram must be a tree: that
fact is not assumed anywhere below, only checked a posteriori as a sanity
check (see `_forest_pruned_spherical_count` at the bottom).

What we DO exploit is permutation symmetry: relabelling the N vertices by
any permutation sigma turns a diagram M into an isomorphic diagram M^sigma
that is spherical together with M or not at all (`is_spherical` depends
only on the isomorphism class, since permuting rows/columns of a symmetric
matrix by sigma is a congruence A -> P A P^T with P a permutation matrix,
which preserves positive-definiteness). So instead of testing all
4^C(N,2) labelled diagrams (~1.07e9 at N=6), it suffices to test one
representative per isomorphism class and weight the result by orbit size.

Isomorphism classes are generated inductively (`enumerate_up_to_isomorphism`):
every class on k vertices arises by extending some class on k-1 vertices
with one new vertex, so extending EVERY (k-1)-representative by EVERY
possible set of edges to the new vertex is guaranteed to hit every k-vertex
class at least once; deduplicating by the canonical code from coxeter.py
(`canonical_code_and_automorphisms`) keeps exactly one representative per
class. This is a direct, if unsophisticated, instance of the standard
"isomorph-free exhaustive generation" technique (as used by e.g. nauty/geng)
-- entirely correct, just not asymptotically optimal.
"""

from __future__ import annotations

import functools
import itertools
import math
import time
from dataclasses import dataclass, field

import numpy as np

from coxeter import (
    LABELS,
    canonical_codes_and_automorphisms_batch,
    diagram_from_edges,
    diagram_to_digit_vector,
    digit_vector_to_diagram,
    is_spherical,
    pair_index_list,
)


# --------------------------------------------------------------------------- #
# Isomorph-free generation of all labelled diagrams on n vertices
# --------------------------------------------------------------------------- #

@functools.lru_cache(maxsize=None)
def _augmentation_layout(k: int) -> tuple[tuple[int, ...], tuple[int, ...], np.ndarray]:
    """Where, in the flat pair_index_list(k) vector, the C(k-1,2) "old" pair
    values and the (k-1) "new vertex" pair values land -- plus a
    precomputed grid of all 4^{k-1} possible new-vertex label choices
    (digits 0..3), shape (4**(k-1), k-1). All three depend only on k, so
    this is computed once and cached.
    """
    pairs_k = pair_index_list(k)
    pos = {p: t for t, p in enumerate(pairs_k)}
    old_positions = tuple(pos[p] for p in pair_index_list(k - 1))
    new_positions = tuple(pos[(i, k - 1)] for i in range(k - 1))
    if k - 1 == 0:
        grid = np.zeros((1, 0), dtype=np.int64)
    else:
        grid = np.array(list(itertools.product(range(4), repeat=k - 1)), dtype=np.int64)
    return old_positions, new_positions, grid


def enumerate_up_to_isomorphism(n: int) -> dict[int, tuple[np.ndarray, int]]:
    """One representative per isomorphism class of labelled diagrams on n
    vertices, built inductively (see module docstring).

    Returns a dict: canonical_code -> (representative_matrix, aut_count).
    `orbit_size = n! // aut_count` is the number of distinct vertex-labelled
    diagrams that representative stands for.

    For every representative at k-1 vertices, ALL 4^{k-1} extensions by a
    new vertex are built and canonicalised in a single batched call (see
    canonical_codes_and_automorphisms_batch) rather than one Python-level
    call per candidate -- this is what makes k = 6 tractable in seconds
    rather than minutes.
    """
    # classes on 0 vertices: a single, trivial empty diagram.
    classes: dict[int, tuple[np.ndarray, int]] = {0: (np.zeros((0, 0), dtype=np.int64), 1)}

    # Chunk many parents together per batched call: this amortises the fixed
    # numpy/Python call overhead across a much larger matmul, which matters
    # once the number of parents (|classes| at k-1) reaches the thousands.
    MAX_ROWS_PER_CHUNK = 20_000

    for k in range(1, n + 1):
        old_positions, new_positions, grid = _augmentation_layout(k)
        m = k * (k - 1) // 2
        B = grid.shape[0]
        parents = [
            diagram_to_digit_vector(rep) if k > 1 else np.zeros(0, dtype=np.int64)
            for rep, _aut in classes.values()
        ]
        parents_per_chunk = max(1, MAX_ROWS_PER_CHUNK // max(B, 1))

        seen: dict[int, np.ndarray] = {}  # code -> full digit vector (length m)
        for c0 in range(0, len(parents), parents_per_chunk):
            chunk = parents[c0 : c0 + parents_per_chunk]
            n_par = len(chunk)
            full_vecs = np.empty((n_par * B, m), dtype=np.int64)
            if new_positions:
                full_vecs[:, new_positions] = np.tile(grid, (n_par, 1))
            if old_positions:
                old_block = np.repeat(np.stack(chunk), B, axis=0)  # (n_par*B, len(old_positions))
                full_vecs[:, old_positions] = old_block

            codes, _ = canonical_codes_and_automorphisms_batch(full_vecs, k)
            uniq_codes, first_idx = np.unique(codes, return_index=True)
            for code, idx in zip(uniq_codes.tolist(), first_idx.tolist()):
                if code not in seen:
                    seen[code] = full_vecs[idx]

        classes = {
            code: (digit_vector_to_diagram(vec, k), 0)  # aut filled in below
            for code, vec in seen.items()
        }
        # aut counts for the final representatives (cheap: only |classes| of them)
        reps_vecs = np.stack([diagram_to_digit_vector(M) for M, _ in classes.values()]) if classes else np.zeros((0, m), dtype=np.int64)
        if len(classes) > 0:
            _, auts = canonical_codes_and_automorphisms_batch(reps_vecs, k)
            classes = {
                code: (M, int(a))
                for (code, (M, _)), a in zip(classes.items(), auts.tolist())
            }

    return classes


@dataclass
class ExhaustiveResult:
    n: int
    n_isomorphism_classes: int
    n_labelled_diagrams: int          # sum of orbit sizes; must equal 4**C(n,2)
    n_spherical_classes: int
    n_spherical_labelled: int         # sum of orbit sizes of spherical classes
    spherical_representatives: list = field(default_factory=list)  # list[np.ndarray]
    elapsed_seconds: float = 0.0


def exhaustive_spherical_search(n: int, keep_representatives: bool = True) -> ExhaustiveResult:
    """Test every labelled diagram on n vertices for the spherical condition,
    via one representative per isomorphism class (see module docstring).

    Every diagram is genuinely tested (via its representative) -- diagrams
    with cycles are not skipped or assumed non-spherical.
    """
    t0 = time.perf_counter()
    classes = enumerate_up_to_isomorphism(n)

    n_labelled = 0
    n_spherical_classes = 0
    n_spherical_labelled = 0
    reps_found = []

    fact_n = math.factorial(n)
    for rep, aut in classes.values():
        orbit = fact_n // aut
        n_labelled += orbit
        if is_spherical(rep):
            n_spherical_classes += 1
            n_spherical_labelled += orbit
            if keep_representatives:
                reps_found.append(rep)

    expected_total = len(LABELS) ** (n * (n - 1) // 2)
    assert n_labelled == expected_total, (
        f"orbit-counting sanity check failed at n={n}: "
        f"sum of orbit sizes = {n_labelled}, expected {expected_total}"
    )

    return ExhaustiveResult(
        n=n,
        n_isomorphism_classes=len(classes),
        n_labelled_diagrams=n_labelled,
        n_spherical_classes=n_spherical_classes,
        n_spherical_labelled=n_spherical_labelled,
        spherical_representatives=reps_found,
        elapsed_seconds=time.perf_counter() - t0,
    )


# --------------------------------------------------------------------------- #
# Cross-check against the earlier, forest-pruned implementation
# --------------------------------------------------------------------------- #
#
# The previous version of this file pruned every diagram whose underlying
# graph has a cycle, using the fact (not assumed here) that such a diagram
# cannot be spherical. It is kept below, clearly separated, purely so the
# two independent methods can be checked against each other: if they ever
# disagree on n_spherical_labelled, one of the two implementations is wrong.

def _forest_pruned_spherical_count(n: int) -> int:
    """Reference count using the tree/forest fact -- NOT used by
    exhaustive_spherical_search above, only for cross-validation below."""
    from coxeter import UnionFind, pair_index_list

    pairs = pair_index_list(n)
    m = len(pairs)
    total = 0
    for mask in range(1 << m):
        edges = []
        uf = UnionFind(n)
        acyclic = True
        for k in range(m):
            if (mask >> k) & 1:
                i, j = pairs[k]
                if not uf.union(i, j):
                    acyclic = False
                    break
                edges.append((i, j))
        if not acyclic:
            continue
        for label_choice in itertools.product((3, 4, 6), repeat=len(edges)):
            M = diagram_from_edges(n, [(i, j, lab) for (i, j), lab in zip(edges, label_choice)])
            if is_spherical(M):
                total += 1
    return total


if __name__ == "__main__":
    for n in range(1, 7):
        res = exhaustive_spherical_search(n, keep_representatives=(n <= 5))
        print(
            f"N={n}: isomorphism classes={res.n_isomorphism_classes:>6,}  "
            f"labelled diagrams (all, incl. cyclic)={res.n_labelled_diagrams:>12,}  "
            f"spherical classes={res.n_spherical_classes:>4,}  "
            f"spherical labelled={res.n_spherical_labelled:>5,}  "
            f"({res.elapsed_seconds:.2f}s)"
        )

    print("\nCross-checking against the forest-pruned reference count (uses the tree fact)...")
    for n in range(1, 6):  # N=6 alone (2^15 masks in pure Python) is the slow part of the old method
        ref = _forest_pruned_spherical_count(n)
        new = exhaustive_spherical_search(n, keep_representatives=False).n_spherical_labelled
        status = "OK" if ref == new else "MISMATCH"
        print(f"  N={n}: forest-pruned={ref:>5,}  isomorphism-based={new:>5,}  [{status}]")
        assert ref == new
    print("exhaustive.py: cross-check passed -- both methods agree.")

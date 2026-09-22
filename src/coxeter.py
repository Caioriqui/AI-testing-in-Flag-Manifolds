"""
coxeter.py -- Coxeter diagrams, Cartan-type matrices and the spherical test.

Representation
--------------
A labelled diagram on N vertices is a symmetric integer numpy array `M` of
shape (N, N): `M[i, i] = 0` (unused), and for i != j, `M[i, j] == M[j, i]`
takes a value in {2, 3, 4, 6}:

    2  ->  no edge        (m_ij = 2, s_i and s_j commute)
    3  ->  edge, unlabelled ("3" is conventionally omitted when drawing)
    4  ->  edge labelled 4
    6  ->  edge labelled 6

Restricting labels to this alphabet of size 4 reaches every spherical type
except H3, H4 and I2(m >= 5), which require a label >= 5 (see report,
section 1.2). This is a project-wide convention: every other module in this
package assumes labels are drawn from exactly these four values.

Associated to a diagram is the Cartan-type (Gram) matrix

    A_ii = 2,   A_ij = -2 cos(pi / m_ij)  (i != j),

and the diagram is *spherical* iff A is positive definite (Coxeter's
theorem). Positive-definiteness is what "5" and "6" edges is spherical
depends on, so `is_spherical` is the single source of truth used by every
other module (exhaustive search, data generation, RL reward).
"""
from __future__ import annotations

import functools
import itertools
import math
from typing import Iterable, Sequence

import numpy as np

LABELS: tuple[int, ...] = (2, 3, 4, 6)  # 2 == "no edge"
NO_EDGE = 2

# Labels as 2-bit "digits" 0..3, used only by the permutation-symmetry code
# below (canonical_code_and_automorphisms), where diagrams need to be packed
# into a single base-4 integer.
_LABEL_TO_DIGIT = {lab: d for d, lab in enumerate(LABELS)}
_DIGIT_TO_LABEL = {d: lab for lab, d in _LABEL_TO_DIGIT.items()}


# --------------------------------------------------------------------------- #
# Basic diagram construction
# --------------------------------------------------------------------------- #

def num_pairs(n: int) -> int:
    """C(n, 2), the number of unordered vertex pairs."""
    return n * (n - 1) // 2


def pair_index_list(n: int) -> list[tuple[int, int]]:
    """Canonical ordering of the C(n, 2) unordered pairs of {0, ..., n-1}.

    Every array of "one value per pair" in this package (label choices,
    edge subsets, ...) is indexed according to this ordering, so it must be
    used consistently by any code building or reading such arrays.
    """
    return list(itertools.combinations(range(n), 2))


def empty_diagram(n: int) -> np.ndarray:
    """The diagram on n vertices with no edges at all."""
    M = np.full((n, n), NO_EDGE, dtype=np.int64)
    np.fill_diagonal(M, 0)
    return M


def labels_to_matrix(n: int, label_values: Sequence[int]) -> np.ndarray:
    """Build the symmetric (n, n) label matrix from values on pair_index_list(n).

    `label_values[k]` is the label of the k-th pair in `pair_index_list(n)`.
    """
    pairs = pair_index_list(n)
    if len(label_values) != len(pairs):
        raise ValueError(
            f"expected {len(pairs)} label values for n={n}, got {len(label_values)}"
        )
    M = empty_diagram(n)
    for (i, j), lab in zip(pairs, label_values):
        M[i, j] = M[j, i] = lab
    return M


def diagram_from_edges(n: int, edges: Iterable[tuple[int, int, int]]) -> np.ndarray:
    """Build a diagram from an explicit list of (i, j, label) triples.

    Any pair not mentioned is assumed to have `label = NO_EDGE`.
    """
    M = empty_diagram(n)
    for i, j, lab in edges:
        M[i, j] = M[j, i] = lab
    return M


# --------------------------------------------------------------------------- #
# Cartan-type matrix and the spherical test
# --------------------------------------------------------------------------- #

def cartan_matrix(M: np.ndarray) -> np.ndarray:
    """Cartan-type (Gram) matrix A_ij = 2 (i=j), -2cos(pi/m_ij) (i!=j).

    The diagonal of M is unused (by convention 0) and would otherwise cause
    a division by zero in pi / m_ij, so it is masked out before the divide.
    """
    Mf = M.astype(np.float64).copy()
    np.fill_diagonal(Mf, 1.0)  # placeholder, overwritten by fill_diagonal below
    A = -2.0 * np.cos(np.pi / Mf)
    np.fill_diagonal(A, 2.0)
    return A


def is_spherical(M: np.ndarray, tol: float = 1e-8) -> bool:
    """Whether the Cartan-type matrix of M is positive definite.

    Positive *semi*-definite-but-singular diagrams (e.g. affine cycles) are
    correctly reported as NOT spherical: `tol` is a strictly-positive
    threshold on the smallest eigenvalue, not a numerical slack around 0.
    """
    A = cartan_matrix(M)
    eigvals = np.linalg.eigvalsh(A)
    return bool(eigvals.min() > tol)


def spherical_margin(M: np.ndarray) -> float:
    """Smallest eigenvalue of the Cartan-type matrix.

    Useful as a continuous reward/diagnostic signal beyond the bare pass/fail
    of `is_spherical` (e.g. for reward shaping in the RL agent, part 3).
    """
    A = cartan_matrix(M)
    return float(np.linalg.eigvalsh(A).min())


def spherical_margin_and_test(M: np.ndarray, tol: float = 1e-8) -> tuple[float, bool]:
    """(spherical_margin(M), is_spherical(M, tol)), from a SINGLE eigen-
    decomposition.

    `is_spherical` and `spherical_margin` each call `cartan_matrix` +
    `np.linalg.eigvalsh` independently, so a caller needing both values --
    e.g. env.py's `step`, once per environment per environment-step, the
    hottest loop in the whole package -- would otherwise pay for that
    decomposition twice per call for no reason. The semantics here are
    exactly those of calling both functions separately with the same
    `tol`; this is a performance shortcut, not a new definition, and
    `is_spherical`/`spherical_margin` remain the source of truth for every
    non-hot-loop caller (exhaustive search, data generation, analysis).
    """
    A = cartan_matrix(M)
    margin = float(np.linalg.eigvalsh(A).min())
    return margin, margin > tol


# --------------------------------------------------------------------------- #
# Graph utilities (union-find, components, acyclicity)
# --------------------------------------------------------------------------- #

class UnionFind:
    """Minimal union-find (disjoint-set) structure with path halving."""

    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, x: int, y: int) -> bool:
        """Merge the components of x and y. Returns False if already joined
        (i.e. the edge (x, y) would close a cycle)."""
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return False
        self.parent[rx] = ry
        return True


def edge_list(M: np.ndarray) -> list[tuple[int, int]]:
    """List of (i, j), i < j, present in M (label != NO_EDGE)."""
    n = M.shape[0]
    return [(i, j) for i in range(n) for j in range(i + 1, n) if M[i, j] != NO_EDGE]


def num_edges(M: np.ndarray) -> int:
    return len(edge_list(M))


def connected_components(M: np.ndarray) -> int:
    """Number of connected components of the underlying (unlabelled) graph."""
    n = M.shape[0]
    uf = UnionFind(n)
    for i, j in edge_list(M):
        uf.union(i, j)
    return len({uf.find(v) for v in range(n)})


def is_acyclic(M: np.ndarray) -> bool:
    """Whether the underlying graph is a forest (no cycles).

    Every connected spherical diagram is a tree, so this is a cheap
    necessary condition used throughout the package to prune the search
    space before ever computing an eigenvalue.
    """
    n = M.shape[0]
    uf = UnionFind(n)
    for i, j in edge_list(M):
        if not uf.union(i, j):
            return False
    return True


# --------------------------------------------------------------------------- #
# Permutation symmetry (S_n acting simultaneously on rows and columns)
# --------------------------------------------------------------------------- #
#
# A relabelling of the n vertices by a permutation sigma sends a diagram M to
# M^sigma with M^sigma[i, j] = M[sigma(i), sigma(j)]; two diagrams related
# this way are the "same" diagram drawn with a different vertex order, so
# they are spherical together or not at all. Exploiting this action is what
# lets exhaustive search reduce the ~n! copies of a diagram inherent to any
# labelling by vertex to a single representative, WITHOUT assuming anything
# about which diagrams are spherical (contrast with the acyclic/forest
# pruning above, which uses that fact and is therefore not used here).
#
# Implementation note: for each n we precompute, once, how every one of the
# n! permutations reorders the flat vector of the C(n,2) pair-values (in
# pair_index_list(n) order). Applying a permutation to a candidate diagram
# is then a single fancy-index lookup into that fixed table rather than n!
# separate matrix rearrangements, which is what makes checking millions of
# candidates at n = 6 practical.

@functools.lru_cache(maxsize=None)
def _permutation_reorder_table(n: int) -> np.ndarray:
    """Table of shape (n!, C(n,2)): row r gives, for each pair position t in
    pair_index_list(n), the position in that same list that the r-th
    permutation of {0,...,n-1} reads its value from.

    That is, if `vec` holds M's values in pair_index_list(n) order, then
    `vec[table[r]]` holds M^perm_r's values in that same order.
    """
    pairs = pair_index_list(n)
    pair_to_idx = {p: t for t, p in enumerate(pairs)}
    rows = []
    for perm in itertools.permutations(range(n)):
        row = []
        for (i, j) in pairs:
            pi, pj = perm[i], perm[j]
            if pi > pj:
                pi, pj = pj, pi
            row.append(pair_to_idx[(pi, pj)])
        rows.append(row)
    return np.array(rows, dtype=np.int64)


def diagram_to_digit_vector(M: np.ndarray) -> np.ndarray:
    """M's C(n,2) pair-values, in pair_index_list(n) order, mapped to digits
    0..3 (so they can be packed into a single base-4 integer)."""
    n = M.shape[0]
    pairs = pair_index_list(n)
    return np.array([_LABEL_TO_DIGIT[int(M[i, j])] for i, j in pairs], dtype=np.int64)


def digit_vector_to_diagram(vec: np.ndarray, n: int) -> np.ndarray:
    """Inverse of diagram_to_digit_vector."""
    M = empty_diagram(n)
    for (i, j), d in zip(pair_index_list(n), vec.tolist()):
        M[i, j] = M[j, i] = _DIGIT_TO_LABEL[int(d)]
    return M


@functools.lru_cache(maxsize=None)
def _canonical_code_weight_matrix(n: int) -> np.ndarray:
    """Matrix W of shape (C(n,2), n!) such that, for a digit vector `vec`,
    `vec @ W` gives, in one matrix product, the base-4 code of `vec` under
    every one of the n! relabellings -- avoiding ever materialising the
    (batch, n!, C(n,2)) tensor that a naive `vec[table]` lookup would need.

    Derivation: the code of the p-th relabelling is
    sum_t vec[table[p, t]] * powers[t]. Substituting s = table[p, t] (a
    bijection on {0,...,m-1} since row p of `table` is a permutation) gives
    sum_s vec[s] * powers[table_inv[p, s]], i.e. W[s, p] = powers[table_inv[p, s]]
    with table_inv the row-wise inverse permutation of `table`.
    """
    table = _permutation_reorder_table(n)                  # (n!, m)
    m = table.shape[1]
    table_inv = np.argsort(table, axis=1)                   # (n!, m)
    powers = (4 ** np.arange(m - 1, -1, -1)).astype(np.int64)
    W = powers[table_inv].T.copy()                          # (m, n!)
    return W


def canonical_codes_and_automorphisms_batch(
    vecs: np.ndarray, n: int, max_rows_per_chunk: int = 20_000
) -> tuple[np.ndarray, np.ndarray]:
    """Batched version: `vecs` has shape (B, C(n,2)), each row a diagram's
    digit vector (diagram_to_digit_vector order). Returns (codes, aut_counts),
    each of shape (B,) -- see canonical_code_and_automorphisms for semantics.

    Processing a batch in one shot (rather than one diagram at a time) is
    what makes generating all isomorphism classes at n = 6 practical: a
    single (B, m) @ (m, n!) matrix product computes the code of every
    candidate under every one of the n! relabellings at once. The (B, n!)
    intermediate is chunked along B (`max_rows_per_chunk`) so B itself can
    run into the millions (as it does for the n = 6 representative list)
    without the intermediate ever exceeding a few hundred MB.
    """
    W = _canonical_code_weight_matrix(n)               # (m, n!)
    B = vecs.shape[0]
    if B <= max_rows_per_chunk:
        codes_all = vecs @ W                            # (B, n!)
        min_codes = codes_all.min(axis=1)
        aut_counts = (codes_all == min_codes[:, None]).sum(axis=1)
        return min_codes, aut_counts

    min_codes = np.empty(B, dtype=np.int64)
    aut_counts = np.empty(B, dtype=np.int64)
    for s in range(0, B, max_rows_per_chunk):
        e = min(s + max_rows_per_chunk, B)
        codes_chunk = vecs[s:e] @ W                     # (chunk, n!)
        mc = codes_chunk.min(axis=1)
        min_codes[s:e] = mc
        aut_counts[s:e] = (codes_chunk == mc[:, None]).sum(axis=1)
    return min_codes, aut_counts


def canonical_code_and_automorphisms(M: np.ndarray) -> tuple[int, int]:
    """Isomorphism invariant of M under simultaneous row/column permutation,
    plus |Aut(M)|, the number of permutations fixing M exactly.

    The invariant is the smallest base-4 integer among the C(n,2)-digit
    codes of every one of the n! relabellings of M -- i.e. it depends only
    on the isomorphism class of M, so two diagrams are isomorphic iff this
    returns the same code for both. |Aut(M)| is the number of permutations
    achieving that minimum, which (standard orbit-stabiliser argument) is
    the same for every element of the orbit, in particular for M itself.

    Single-diagram convenience wrapper around the batched version below.
    """
    n = M.shape[0]
    vec = diagram_to_digit_vector(M)[None, :]         # (1, C(n,2))
    codes, auts = canonical_codes_and_automorphisms_batch(vec, n)
    return int(codes[0]), int(auts[0])


def orbit_size(M: np.ndarray) -> int:
    """Number of distinct vertex-labelled diagrams isomorphic to M, i.e.
    n! / |Aut(M)|."""
    n = M.shape[0]
    _, aut_count = canonical_code_and_automorphisms(M)
    return math.factorial(n) // aut_count


# --------------------------------------------------------------------------- #
# Sanity checks -- explicit, known examples (run this file directly)
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    # A_3: path on 4 vertices, all edges labelled 3 -> finite type, spherical.
    A3 = diagram_from_edges(4, [(0, 1, 3), (1, 2, 3), (2, 3, 3)])
    assert is_spherical(A3), "A_3 (path, all labels 3) must be spherical"

    # Affine A~_2: 3-cycle, all edges labelled 3 -> positive SEMI-definite,
    # singular, must be reported as NOT spherical.
    A2_affine = diagram_from_edges(3, [(0, 1, 3), (1, 2, 3), (0, 2, 3)])
    assert not is_spherical(A2_affine), "affine A~_2 (triangle) must not be spherical"
    assert not is_acyclic(A2_affine), "a 3-cycle is not acyclic"
    assert abs(spherical_margin(A2_affine)) < 1e-8, "affine diagram must be singular"

    # D_4: star with 3 edges from a common centre, all labelled 3 -> spherical.
    D4 = diagram_from_edges(4, [(0, 1, 3), (0, 2, 3), (0, 3, 3)])
    assert is_spherical(D4), "D_4 (claw, all labels 3) must be spherical"
    assert is_acyclic(D4)
    assert connected_components(D4) == 1

    # B_2/C_2: single edge labelled 4 -> spherical (dihedral group of order 8).
    B2 = diagram_from_edges(2, [(0, 1, 4)])
    assert is_spherical(B2), "B_2 (single edge, label 4) must be spherical"

    # G_2: single edge labelled 6 -> spherical (dihedral group of order 12).
    G2 = diagram_from_edges(2, [(0, 1, 6)])
    assert is_spherical(G2), "G_2 (single edge, label 6) must be spherical"

    # Disjoint union A_1 x A_1 (no edge at all on 2 vertices) -> spherical,
    # 2 connected components.
    disjoint = empty_diagram(2)
    assert is_spherical(disjoint)
    assert connected_components(disjoint) == 2

    # A_1 x A_2, packaged with A_2 as a 2-cycle worth of dense labels, must
    # fail to be spherical once the underlying graph has a cycle at all,
    # regardless of the specific labels chosen.
    dense_cycle = diagram_from_edges(3, [(0, 1, 3), (1, 2, 4), (0, 2, 6)])
    assert not is_acyclic(dense_cycle)
    assert not is_spherical(dense_cycle)

    # Permutation-symmetry checks. Two diagrams that only differ by a vertex
    # relabelling must get the same canonical code, and a different aut
    # count from a genuinely inequivalent diagram would be a red flag.
    path_012 = diagram_from_edges(3, [(0, 1, 3), (1, 2, 4)])   # A2--o--o (label 4)
    path_relabelled = diagram_from_edges(3, [(2, 0, 3), (0, 1, 4)])  # same path, vertices renamed
    code1, aut1 = canonical_code_and_automorphisms(path_012)
    code2, aut2 = canonical_code_and_automorphisms(path_relabelled)
    assert code1 == code2, "relabelled copies of the same diagram must share a canonical code"
    assert aut1 == aut2 == 1, "an asymmetric path (labels 3 then 4) has a trivial automorphism group"
    assert orbit_size(path_012) == math.factorial(3) // 1 == 6

    # A single edge on 2 vertices, by contrast, IS fixed by swapping its two
    # endpoints (the label doesn't care which side is which), so its
    # automorphism group has order 2 and its orbit has size 2!/2 = 1.
    single_edge = diagram_from_edges(2, [(0, 1, 3)])
    _, aut_edge = canonical_code_and_automorphisms(single_edge)
    assert aut_edge == 2
    assert orbit_size(single_edge) == 1

    # D_4's centre is fixed by any permutation of its 3 leaves: |Aut(D_4)| = 3!
    _, aut_d4 = canonical_code_and_automorphisms(D4)
    assert aut_d4 == 6
    assert orbit_size(D4) == math.factorial(4) // 6 == 4

    # spherical_margin_and_test must agree EXACTLY with calling
    # spherical_margin and is_spherical separately, on both a spherical and
    # a non-spherical example -- it is a performance shortcut, not a
    # different definition.
    for M_check in (A3, A2_affine, D4, dense_cycle):
        margin_combined, success_combined = spherical_margin_and_test(M_check)
        assert margin_combined == spherical_margin(M_check)
        assert success_combined == is_spherical(M_check)
    print("spherical_margin_and_test: agrees with spherical_margin + is_spherical")

    print("coxeter.py: all sanity checks passed.")

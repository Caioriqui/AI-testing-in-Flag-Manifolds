"""
policy.py -- hand-rolled graph neural network actor-critic for
Objective 1 (MM845). No PyTorch Geometric (or any other GNN library) is
used, per Diretrizes.txt: every message-passing step below is plain
PyTorch tensor algebra over the fixed N_MAX = 8 vertex graph.

Because N_MAX is fixed and small, the network never has to handle a
variable number of *tensor* rows/columns, padding shapes, or batching
machinery beyond an ordinary leading batch dimension -- that is exactly
what makes a manual implementation this short reasonable instead of a
false economy. What N_MAX being fixed does NOT mean is that every episode
uses all 8 vertices: env.py samples a variable `n_active <= N_MAX` per
episode and pads the rest with permanently isolated vertices (see env.py's
module docstring for why that padding is invisible to the reward signal).
This module is what makes that padding invisible to the POLICY as well:

    1. `n_actives` (one int per batch element) is now a required second
       input everywhere alongside `states`, because n_active cannot be
       recovered from `states` alone (a real isolated vertex and a padding
       vertex look identical in the raw array).
    2. Message passing masks out messages FROM padding vertices, so they
       cannot influence the hidden state of real vertices.
    3. The value head's pooling is a MASKED mean over real vertices only.
    4. The policy head's logits are masked (-inf) for every action whose
       pair touches a padding vertex, before the Categorical distribution
       is built -- the standard invalid-action-masking technique for
       variable-size combinatorial action spaces, which is what lets this
       whole project keep ONE fixed-size batched action space (env.py's
       "Action space" section) instead of a per-episode variable one.

Architecture
------------
1. Each vertex i owns a learned embedding `node_embed[i]` (an (N_MAX, d)
   lookup table) as its initial hidden state. There is no other per-vertex
   feature: the vertices are told apart only by their fixed position, and
   the action space (env.py) is itself indexed by that same fixed position,
   so this is required, not a simplification. A padding vertex gets the
   same embedding as a real vertex sitting at that index would -- what
   marks it as padding is purely that its outgoing messages are masked to
   zero (point 2 above), not a different embedding table.
2. `n_layers` rounds of message passing. In each round vertex i receives a
   message from every other (non-padding) vertex j, built from j's current
   hidden state and an embedding of the *current* label on edge (i, j);
   messages are summed over j and combined with i's own state through a
   small MLP, with a residual connection (a GRU-style gated update was
   deliberately not used: Diretrizes.txt prefers the conceptually simplest
   option whenever a choice does not bear on training stability, and a
   residual MLP is the simpler of the two while still avoiding vanishing
   updates across layers, which is a stability property in its own right).
3. Two heads read the final hidden states:
   - policy head: for every (unordered pair, label) combination, an MLP
     scores concat(h_i, h_j, label_embedding) -> one logit. The resulting
     (num_pairs, 4) logit grid is flattened in exactly the order
     `encode_action` / `decode_action` (env.py) expect -- pair_idx varies
     slower than label_idx -- so the two files can never disagree about
     which logit means which action. Logits for pairs touching a padding
     vertex are then overwritten with -inf (point 4 above).
   - value head: an MLP over the mean-pooled hidden state of REAL vertices
     only -> scalar V(s).

This module has no notion of DynkinEnv beyond importing the fixed pair
ordering and action count from it (single source of truth, as env.py's own
module docstring asks for).
"""
from __future__ import annotations

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

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from src.coxeter import LABELS
from src.env import N_MAX, num_actions, pairs_for

_MAX_LABEL_VALUE = max(LABELS)  # 6 -- diagonal entries (0) are masked out anyway
_NEG_INF = float("-inf")


class DynkinGNN(nn.Module):
    """Actor-critic over labelled diagrams on a fixed N_MAX-vertex graph,
    with a variable number `n_active <= N_MAX` of real vertices per
    episode (see module docstring)."""

    def __init__(
        self,
        n_max: int = N_MAX,
        hidden_dim: int = 64,
        label_embed_dim: int = 8,
        n_layers: int = 3,
    ):
        super().__init__()
        self.n_max = n_max
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        pairs = pairs_for(n_max)
        self.num_pairs = len(pairs)
        self.num_actions = num_actions(n_max)
        # Buffers (not parameters): fixed index tensors reused every forward pass.
        self.register_buffer("idx_i", torch.tensor([p[0] for p in pairs], dtype=torch.long))
        self.register_buffer("idx_j", torch.tensor([p[1] for p in pairs], dtype=torch.long))
        self.register_buffer("diag_mask", torch.eye(n_max, dtype=torch.bool))
        # arange(n_max) reused every forward pass to build the active-vertex
        # mask from n_actives via broadcasting comparison (see _active_mask).
        self.register_buffer("_vertex_range", torch.arange(n_max, dtype=torch.long))

        # Map a raw label value (0 on the diagonal, or one of LABELS) to an
        # index in [0, len(LABELS)); the diagonal's value is irrelevant
        # since its message is masked out below, so it is left at 0.
        label_to_idx = torch.zeros(_MAX_LABEL_VALUE + 1, dtype=torch.long)
        for idx, lab in enumerate(LABELS):
            label_to_idx[lab] = idx
        self.register_buffer("label_to_idx", label_to_idx)

        self.node_embed = nn.Embedding(n_max, hidden_dim)
        self.label_embed = nn.Embedding(len(LABELS), label_embed_dim)

        self.message_fn = nn.ModuleList(
            [nn.Linear(hidden_dim + label_embed_dim, hidden_dim) for _ in range(n_layers)]
        )
        self.update_fn = nn.ModuleList(
            [nn.Linear(2 * hidden_dim, hidden_dim) for _ in range(n_layers)]
        )

        self.policy_head = nn.Sequential(
            nn.Linear(2 * hidden_dim + label_embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def _as_tensor(self, x, dtype, device) -> torch.Tensor:
        if not torch.is_tensor(x):
            return torch.as_tensor(x, dtype=dtype, device=device)
        return x.to(dtype=dtype, device=device)

    def _active_mask(self, n_actives: torch.Tensor) -> torch.Tensor:
        """n_actives: (B,) long. Returns (B, n_max) bool: True where the
        vertex index is a REAL vertex of that batch element (< n_active),
        False where it is padding."""
        return self._vertex_range.unsqueeze(0) < n_actives.unsqueeze(1)

    def _encode(self, states: torch.Tensor, active_mask: torch.Tensor) -> torch.Tensor:
        """states: (B, N_MAX, N_MAX) long, label values. active_mask: (B,
        N_MAX) bool. Returns final node hidden states h: (B, N_MAX,
        hidden_dim) -- hidden states of padding vertices are NOT meaningful
        (nothing downstream reads them without also consulting
        active_mask), only guaranteed to never leak into real vertices.
        """
        B = states.shape[0]
        label_idx = self.label_to_idx[states]  # (B, N_MAX, N_MAX)
        edge_emb = self.label_embed(label_idx)  # (B, N_MAX, N_MAX, label_dim)

        h = self.node_embed.weight.unsqueeze(0).expand(B, -1, -1)  # (B, N_MAX, hidden)

        diag_mask = self.diag_mask.view(1, self.n_max, self.n_max, 1)  # broadcast over B, hidden
        # inactive_j[:, i, j, :] is True iff vertex j is padding, for every i:
        # a message from a padding vertex j must never reach any i.
        inactive_j = (~active_mask).view(B, 1, self.n_max, 1)

        for layer in range(self.n_layers):
            h_j = h.unsqueeze(1).expand(B, self.n_max, self.n_max, self.hidden_dim)  # h_j[:,i,j,:]=h[:,j,:]
            msg_in = torch.cat([h_j, edge_emb], dim=-1)  # (B, N_MAX, N_MAX, hidden+label_dim)
            msgs = F.relu(self.message_fn[layer](msg_in))  # (B, N_MAX, N_MAX, hidden)
            msgs = msgs.masked_fill(diag_mask, 0.0)  # no self-message
            msgs = msgs.masked_fill(inactive_j, 0.0)  # no message out of a padding vertex
            agg = msgs.sum(dim=2)  # (B, N_MAX, hidden) -- sum over j

            upd_in = torch.cat([h, agg], dim=-1)  # (B, N_MAX, 2*hidden)
            h = F.relu(h + self.update_fn[layer](upd_in))  # residual update

        return h

    def forward(self, states, n_actives) -> tuple[torch.Tensor, torch.Tensor]:
        """states: array-like (B, N_MAX, N_MAX) of label values (numpy or
        torch). n_actives: array-like (B,) of ints, the number of real
        vertices for each batch element (cannot be derived from `states`,
        see module docstring). Returns (logits, value): logits (B,
        num_actions) with -inf on every action touching a padding vertex,
        value (B,)."""
        device = self.node_embed.weight.device
        states = self._as_tensor(states, torch.long, device)
        n_actives = self._as_tensor(n_actives, torch.long, device)

        active_mask = self._active_mask(n_actives)  # (B, N_MAX) bool
        h = self._encode(states, active_mask)  # (B, N_MAX, hidden)
        B = h.shape[0]

        h_i = h[:, self.idx_i, :]  # (B, num_pairs, hidden)
        h_j = h[:, self.idx_j, :]  # (B, num_pairs, hidden)
        pair_feat = torch.cat([h_i, h_j], dim=-1)  # (B, num_pairs, 2*hidden)
        pair_feat = pair_feat.unsqueeze(2).expand(B, self.num_pairs, len(LABELS), -1)

        label_emb_all = self.label_embed.weight.view(1, 1, len(LABELS), -1)
        label_emb_all = label_emb_all.expand(B, self.num_pairs, len(LABELS), -1)

        head_in = torch.cat([pair_feat, label_emb_all], dim=-1)
        logits = self.policy_head(head_in).squeeze(-1)  # (B, num_pairs, len(LABELS))
        logits = logits.reshape(B, self.num_pairs * len(LABELS))  # matches encode_action order

        # A pair is valid iff BOTH its vertices are real; every label choice
        # for an invalid pair is masked identically (a pair's validity does
        # not depend on the label being written).
        pair_active = active_mask[:, self.idx_i] & active_mask[:, self.idx_j]  # (B, num_pairs)
        action_active = pair_active.unsqueeze(-1).expand(B, self.num_pairs, len(LABELS))
        action_active = action_active.reshape(B, self.num_pairs * len(LABELS))
        logits = logits.masked_fill(~action_active, _NEG_INF)

        mask_f = active_mask.unsqueeze(-1).to(h.dtype)  # (B, N_MAX, 1)
        pooled = (h * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1.0)  # masked mean, real vertices only
        value = self.value_head(pooled).squeeze(-1)  # (B,)

        return logits, value

    def get_action_and_value(
        self, states, n_actives, action: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Standard PPO helper: sample (or evaluate a given) action, and
        return (action, log_prob, entropy, value). `n_actives` must be the
        SAME per-batch-element values used when the action was originally
        sampled (train_ppo.py threads this through the rollout buffer) --
        using a different mask at update time than at sampling time would
        silently change which actions were "available", corrupting the PPO
        importance ratio."""
        logits, value = self.forward(states, n_actives)
        dist = Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        logprob = dist.log_prob(action)
        entropy = dist.entropy()
        return action, logprob, entropy, value


# --------------------------------------------------------------------------- #
# Sanity checks -- run this file directly (requires torch; not part of the
# numpy-only modules above it in the package)
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    from src.env import DynkinEnv, pairs_for as _pairs_for, decode_action

    torch.manual_seed(0)

    n_max = N_MAX
    model = DynkinGNN(n_max=n_max, hidden_dim=32, label_embed_dim=8, n_layers=2)

    env = DynkinEnv(n_max=n_max, max_steps=10, dataset="A", seed=0)
    results = [env.reset(seed=k) for k in range(5)]
    import numpy as np

    batch = np.stack([s for s, _ in results])  # (5, N_MAX, N_MAX)
    n_actives_full = np.array([n for _, n in results])  # all == n_max (default active_ns)
    assert np.all(n_actives_full == n_max)

    logits, value = model(batch, n_actives_full)
    assert logits.shape == (5, model.num_actions) == (5, 112)
    assert value.shape == (5,)
    assert torch.isfinite(logits).all()  # nothing masked when n_active == n_max
    print("forward shapes OK (full N_MAX, no masking):", logits.shape, value.shape)

    action, logprob, entropy, value2 = model.get_action_and_value(batch, n_actives_full)
    assert action.shape == (5,)
    assert (action >= 0).all() and (action < model.num_actions).all()
    assert logprob.shape == (5,)
    assert entropy.shape == (5,)
    assert torch.allclose(value, value2)
    print("get_action_and_value OK, sampled actions:", action.tolist())

    # Gradients must flow into every parameter group (policy head, value
    # head, message/update MLPs, both embedding tables) from a single
    # combined loss -- if any of them were disconnected from the compute
    # graph, its .grad would stay None.
    loss = logprob.sum() + value2.sum() + entropy.sum()
    loss.backward()
    n_missing_grad = sum(1 for p in model.parameters() if p.grad is None)
    assert n_missing_grad == 0, f"{n_missing_grad} parameters got no gradient"
    print("backward OK: every parameter received a gradient")

    # Evaluating a specific, previously-sampled action must match the
    # sampled log_prob/entropy exactly (both are read off the same logits).
    _, logprob_eval, entropy_eval, _ = model.get_action_and_value(batch, n_actives_full, action=action)
    assert torch.allclose(logprob_eval, logprob)
    assert torch.allclose(entropy_eval, entropy)
    print("action re-evaluation OK")

    # --- Variable N / padding ------------------------------------------------

    env_var = DynkinEnv(n_max=n_max, active_ns=(3, 4, 5), max_steps=10, dataset="A", seed=1)
    results_var = [env_var.reset(seed=k) for k in range(6)]
    batch_var = np.stack([s for s, _ in results_var])
    n_actives_var = np.array([n for _, n in results_var])
    assert set(n_actives_var.tolist()) <= {3, 4, 5}

    logits_var, value_var = model(batch_var, n_actives_var)
    assert logits_var.shape == (6, 112) and value_var.shape == (6,)

    # Every action touching a padding vertex must be masked to -inf, and
    # every action touching only real vertices must be finite.
    pairs8 = _pairs_for(n_max)
    for b in range(6):
        n_act = int(n_actives_var[b])
        for pidx, (i, j) in enumerate(pairs8):
            for lab_idx in range(len(LABELS)):
                a = pidx * len(LABELS) + lab_idx
                is_valid = i < n_act and j < n_act
                is_finite = torch.isfinite(logits_var[b, a]).item()
                assert is_finite == is_valid, (
                    f"batch {b} (n_active={n_act}) action {a} (pair ({i},{j})): "
                    f"expected finite={is_valid}, got finite={is_finite}"
                )
    print("action masking: every logit touching a padding vertex is exactly -inf, others finite")

    # Sampling must never pick a masked (padding) action.
    for _ in range(20):
        actions_var, _, _, _ = model.get_action_and_value(batch_var, n_actives_var)
        for b in range(6):
            n_act = int(n_actives_var[b])
            i, j, _lab = decode_action(int(actions_var[b]), n_max)
            assert i < n_act and j < n_act, f"sampled a padding action for n_active={n_act}"
    print("sampling: never selects an action touching a padding vertex")

    # Padding vertices must not influence real vertices' logits/value: two
    # states that agree on the real (n_active, n_active) submatrix but
    # differ in the padding region (still isolated, just different vertex
    # count nominally -- here we perturb padding-only labels, which should
    # never happen via env.step but is exactly the invariant the mask must
    # enforce against) must give identical logits/value on the real actions
    # and identical value.
    base_state, base_n = results_var[0]
    perturbed = base_state.copy()
    # Rewrite an edge strictly inside the padding block (both endpoints
    # >= base_n); env.py would never do this, but the network's masking
    # must make it harmless regardless.
    pad_i, pad_j = base_n, base_n + 1
    if pad_j < n_max:
        perturbed[pad_i, pad_j] = perturbed[pad_j, pad_i] = 6
        b0 = np.stack([base_state, perturbed])
        n0 = np.array([base_n, base_n])
        logits0, value0 = model(b0, n0)
        assert torch.allclose(logits0[0], logits0[1], equal_nan=False)
        assert torch.allclose(value0[0], value0[1])
        print("padding isolation: perturbing only the padding block leaves logits/value unchanged")

    print("policy.py: all sanity checks passed.")

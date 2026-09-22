"""
policy.py -- hand-rolled graph neural network actor-critic for
Objective 1 (MM845). No PyTorch Geometric (or any other GNN library) is
used, per Diretrizes.txt: every message-passing step below is plain
PyTorch tensor algebra over the fixed N = 8 vertex graph.

Because N is fixed and small, the network never has to handle a variable
number of nodes, padding, or batching machinery beyond an ordinary leading
batch dimension -- that is exactly what makes a manual implementation this
short reasonable instead of a false economy.

Architecture
------------
1. Each vertex i owns a learned embedding `node_embed[i]` (an (N, d) lookup
   table) as its initial hidden state. There is no other per-vertex
   feature: the vertices are told apart only by their fixed position, and
   the action space (env.py) is itself indexed by that same fixed position,
   so this is required, not a simplification.
2. `n_layers` rounds of message passing. In each round vertex i receives a
   message from every other vertex j, built from j's current hidden state
   and an embedding of the *current* label on edge (i, j); messages are
   summed over j and combined with i's own state through a small MLP, with
   a residual connection (a GRU-style gated update was deliberately not
   used: Diretrizes.txt prefers the conceptually simplest option whenever a
   choice does not bear on training stability, and a residual MLP is the
   simpler of the two while still avoiding vanishing updates across
   layers, which is a stability property in its own right).
3. Two heads read the final hidden states:
   - policy head: for every (unordered pair, label) combination, an MLP
     scores concat(h_i, h_j, label_embedding) -> one logit. The resulting
     (num_pairs, 4) logit grid is flattened in exactly the order
     `encode_action` / `decode_action` (env.py) expect -- pair_idx varies
     slower than label_idx -- so the two files can never disagree about
     which logit means which action.
   - value head: an MLP over the mean-pooled hidden state -> scalar V(s).

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
from src.env import N_DEFAULT, num_actions, pairs_for

_MAX_LABEL_VALUE = max(LABELS)  # 6 -- diagonal entries (0) are masked out anyway


class DynkinGNN(nn.Module):
    """Actor-critic over labelled diagrams on a fixed N-vertex graph."""

    def __init__(
        self,
        n: int = N_DEFAULT,
        hidden_dim: int = 64,
        label_embed_dim: int = 8,
        n_layers: int = 3,
    ):
        super().__init__()
        self.n = n
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        pairs = pairs_for(n)
        self.num_pairs = len(pairs)
        self.num_actions = num_actions(n)
        # Buffers (not parameters): fixed index tensors reused every forward pass.
        self.register_buffer("idx_i", torch.tensor([p[0] for p in pairs], dtype=torch.long))
        self.register_buffer("idx_j", torch.tensor([p[1] for p in pairs], dtype=torch.long))
        self.register_buffer("diag_mask", torch.eye(n, dtype=torch.bool))

        # Map a raw label value (0 on the diagonal, or one of LABELS) to an
        # index in [0, len(LABELS)); the diagonal's value is irrelevant
        # since its message is masked out below, so it is left at 0.
        label_to_idx = torch.zeros(_MAX_LABEL_VALUE + 1, dtype=torch.long)
        for idx, lab in enumerate(LABELS):
            label_to_idx[lab] = idx
        self.register_buffer("label_to_idx", label_to_idx)

        self.node_embed = nn.Embedding(n, hidden_dim)
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

    def _encode(self, states: torch.Tensor) -> torch.Tensor:
        """states: (B, N, N) long, label values. Returns final node hidden
        states h: (B, N, hidden_dim)."""
        B = states.shape[0]
        label_idx = self.label_to_idx[states]  # (B, N, N)
        edge_emb = self.label_embed(label_idx)  # (B, N, N, label_dim)

        h = self.node_embed.weight.unsqueeze(0).expand(B, -1, -1)  # (B, N, hidden)

        diag_mask = self.diag_mask.view(1, self.n, self.n, 1)  # broadcast over B, hidden

        for layer in range(self.n_layers):
            h_j = h.unsqueeze(1).expand(B, self.n, self.n, self.hidden_dim)  # h_j[:,i,j,:]=h[:,j,:]
            msg_in = torch.cat([h_j, edge_emb], dim=-1)  # (B, N, N, hidden+label_dim)
            msgs = F.relu(self.message_fn[layer](msg_in))  # (B, N, N, hidden)
            msgs = msgs.masked_fill(diag_mask, 0.0)  # no self-message
            agg = msgs.sum(dim=2)  # (B, N, hidden) -- sum over j

            upd_in = torch.cat([h, agg], dim=-1)  # (B, N, 2*hidden)
            h = F.relu(h + self.update_fn[layer](upd_in))  # residual update

        return h

    def forward(self, states) -> tuple[torch.Tensor, torch.Tensor]:
        """states: array-like (B, N, N) of label values (numpy or torch).
        Returns (logits, value): logits (B, num_actions), value (B,)."""
        device = self.node_embed.weight.device
        if not torch.is_tensor(states):
            states = torch.as_tensor(states, dtype=torch.long, device=device)
        else:
            states = states.long().to(device)
        h = self._encode(states)  # (B, N, hidden)
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

        pooled = h.mean(dim=1)  # (B, hidden)
        value = self.value_head(pooled).squeeze(-1)  # (B,)

        return logits, value

    def get_action_and_value(
        self, states, action: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Standard PPO helper: sample (or evaluate a given) action, and
        return (action, log_prob, entropy, value)."""
        logits, value = self.forward(states)
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
    from src.env import DynkinEnv

    torch.manual_seed(0)

    n = N_DEFAULT
    model = DynkinGNN(n=n, hidden_dim=32, label_embed_dim=8, n_layers=2)

    env = DynkinEnv(n=n, max_steps=10, dataset="A", seed=0)
    states = [env.reset(seed=k) for k in range(5)]
    import numpy as np

    batch = np.stack(states)  # (5, N, N)

    logits, value = model(batch)
    assert logits.shape == (5, model.num_actions) == (5, 112)
    assert value.shape == (5,)
    print("forward shapes OK:", logits.shape, value.shape)

    action, logprob, entropy, value2 = model.get_action_and_value(batch)
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
    _, logprob_eval, entropy_eval, _ = model.get_action_and_value(batch, action=action)
    assert torch.allclose(logprob_eval, logprob)
    assert torch.allclose(entropy_eval, entropy)
    print("action re-evaluation OK")

    print("policy.py: all sanity checks passed.")

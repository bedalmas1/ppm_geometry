"""RLHGNN (A3) model architecture - activity-only next-event prediction.

Adapted from Wang, Yu, Song, Cao, Fan & Zhang, "RLHGNN: Reinforcement
Learning-driven Heterogeneous Graph Neural Network for Next Activity
Prediction in Business Processes," arXiv:2507.02690 (2025), as
re-implemented in https://github.com/Joker3993/RLHGNN. See
paper/related_work_model_audit.md and paper/phase3_baseline_reproduction.md
for the full audit.

**No LICENSE file in the source repo** (checked at clone time, 2026 - the
GitHub repo ships code and a README only). The architecture below is a
from-scratch reimplementation against the public arXiv paper (Sec. IV) and
the repo's own model/model.py, not a verbatim copy of the file - kept
functionally faithful (same layer types, same equations) but restructured to
this project's activity-only scope.

**Scope decisions (see adapter.py's docstring for the graph-construction
side)**:
  1. RL/DQN graph-structure selection is dropped entirely. The paper's own
     ablation (Table V) shows the "Comprehensive" fixed structure (forward +
     backward + repeat_next edges, action=3 in the original repo) performs
     within ~1 GMean point of the full RL policy on average and is the
     richest of the four fixed structures - we always use it, for every
     instance. This keeps the core heterogeneous-GNN architecture and its
     relation-specific aggregation strategy (the paper's central technical
     contribution) while dropping the AutoML wrapper around it, consistent
     with this project's pattern of dropping hyperparameter-search/AutoML
     layers elsewhere in the roster.
  2. Activity-only: the original embeds one feature per raw event-log
     column (activity, resource, plus two engineered/discretized timestamp
     features - inter-event duration and case-progression time). Per this
     project's project-wide activity-only scope decision, only the
     `activity` node feature is embedded; `feature_proj`'s input width is
     therefore `hidden_dim` (one feature) rather than `hidden_dim *
     num_features`.
  3. Single fixed chronological train/val/test split (this project's
     `data.splits`), not the original's 3-fold CV over its own
     random-shuffled internal split - consistent with every other model in
     the roster (spec Sec.6: identical splits across every model).

Hyperparameters (hidden_dim=128, num_layers=2, dropout=0.1) match both the
paper's Sec. V-B ("two-layer HeteroGraphConv architecture ... 128 hidden
units per layer ... dropout regularization (0.1)") and the repo's own
Train.py argparse defaults.

**Graph-level readout discrepancy** (documented, not silently fixed): the
paper's prose (Sec. IV-E) describes the readout as "focusing on the current
activity position within the process instance" (i.e. the last node), but
the repo's actual code takes `dgl.max_nodes(hg, 'h')` - an elementwise max
over ALL node embeddings in the graph, not just the last one. We replicate
the code's actual behavior (max-pool over all nodes), since that is what
produced the paper's reported numbers, not what the prose implies.
"""
from __future__ import annotations

import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import HeteroGraphConv, SAGEConv

EDGE_TYPES = (
    ("node", "forward", "node"),
    ("node", "backward", "node"),
    ("node", "repeat_next", "node"),
)

# Relation-specific GraphSAGE aggregators (paper Eq. 4-6): LSTM for the two
# sequential relations (forward/backward), mean for the pattern relation
# (repeat_next).
_AGGREGATOR_BY_RELATION = {"forward": "lstm", "backward": "lstm", "repeat_next": "mean"}


class RLHGNNActivityOnly(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_classes: int,
        hidden_dim: int = 128,
        dropout: float = 0.1,
        num_layers: int = 2,
    ):
        super().__init__()
        self.activity_embedding = nn.Embedding(vocab_size + 1, hidden_dim)
        self.feature_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.hetero_convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_layers):
            conv_dict = {
                etype: SAGEConv(hidden_dim, hidden_dim, aggregator_type=_AGGREGATOR_BY_RELATION[etype[1]])
                for etype in EDGE_TYPES
            }
            self.hetero_convs.append(HeteroGraphConv(conv_dict, aggregate="mean"))
            self.norms.append(nn.LayerNorm(hidden_dim))
        self.dropout = nn.Dropout(dropout)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        """Matches the original repo's own `init_weights`: normal(std=0.1)
        for the activity embedding, zero for biases. The original also has a
        branch intended to Kaiming-init its Linear layers, but that branch
        checks `'linear' in name.lower()` against `named_parameters()` keys
        - which for `nn.Linear` layers inside an `nn.Sequential` (e.g.
        `feature_proj.0.weight`) never matches, so in the original that
        branch is dead code and those layers keep PyTorch's default init.
        Replicated faithfully here (i.e. NOT adding a working Kaiming-init
        branch), since that default-init behavior is what actually produced
        the original's reported results."""
        for name, param in self.named_parameters():
            if "weight" in name and "activity_embedding" in name:
                nn.init.normal_(param, mean=0.0, std=0.1)
            elif "bias" in name:
                nn.init.constant_(param, 0.0)

    def encode(self, hg: dgl.DGLHeteroGraph) -> torch.Tensor:
        """Returns the graph-level max-pooled node embedding (batch,
        hidden_dim), taken right after the last hetero-conv layer and right
        before the classifier head. This project's z_t extraction point for
        RLHGNN (Phase 4) - architecturally distinct from every other model
        in the roster: it summarizes the WHOLE prefix graph (every node,
        via max-pooling), not just a designated last-position/last-token
        hidden state."""
        h = self.activity_embedding(hg.ndata["activity"].long())
        h = self.feature_proj(h)

        for conv, norm in zip(self.hetero_convs, self.norms):
            residual = h
            h_dict = conv(hg, {"node": h})
            h = h_dict["node"]
            h = norm(h + residual)
            h = F.relu(h)
            h = self.dropout(h)

        with hg.local_scope():
            hg.ndata["h"] = h
            graph_embed = dgl.max_nodes(hg, "h")
        return graph_embed

    def forward(self, hg: dgl.DGLHeteroGraph) -> torch.Tensor:
        graph_embed = self.encode(hg)
        return self.classifier(graph_embed)

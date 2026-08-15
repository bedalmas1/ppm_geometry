"""Adapter: this project's common event-log schema -> RLHGNN's expected
heterogeneous prefix graphs.

Reuses the shared next-activity prefix definition
(data.prefixes.make_next_activity_prefixes) and this project's standard
vocabulary convention (PAD/UNK reserved, built from the TRAIN split only -
see process_transformer/generative_lstm adapters), rather than the original
repo's own encode_map (built from train+val+test combined - a leakage
shortcut we do not replicate, same rationale as every other model).

Graph construction below reimplements
build_graph.build_Bidirect_complex_graph (action 3, "Comprehensive" in the
paper) from the original repo - the ONE fixed graph structure this project
uses for every instance (see model.py's docstring for why the RL/DQN
structure-selection stage is dropped). Three edge types per prefix of
length n, node set {0, ..., n-1} in prefix order:
  - forward:      (i, i+1) for i in [0, n-2]              (self-loop if n==1)
  - backward:     (i+1, i) for i in [0, n-2]               (reverse of forward)
  - repeat_next:  for every pair of positions (p_i, p_j), i<j, sharing the
                  same activity, an edge from p_i to the successor of p_j
                  (if it exists) AND an edge from p_j to the successor of
                  p_i (always exists, since p_i < p_j) - see
                  `_repeat_next_edges` for the exact port of the original's
                  `ConnectRepeatedActivities` / `get_index_of_duplicate_elements`.
Unlike every LSTM/Transformer model in this roster, RLHGNN graphs are NOT
padded to a fixed window: each graph has exactly as many nodes as the
prefix is long, matching the original repo's own (padding-free) design.
"""
from __future__ import annotations

from dataclasses import dataclass

import dgl
import pandas as pd
import torch
from torch.utils.data import Dataset

from data.prefixes import make_next_activity_prefixes

PAD = "[PAD]"
UNK = "[UNK]"

make_prefixes = make_next_activity_prefixes


@dataclass(frozen=True)
class Vocab:
    word_dict: dict[str, int]
    class_dict: dict[str, int]

    @property
    def vocab_size(self) -> int:
        return len(self.word_dict)

    @property
    def num_classes(self) -> int:
        return len(self.class_dict)


def build_vocab(train_prefixes: pd.DataFrame) -> Vocab:
    activities = sorted({a for hist in train_prefixes["history"] for a in hist} | set(train_prefixes["next_act"]))
    word_dict = {PAD: 0, UNK: 1, **{a: i + 2 for i, a in enumerate(activities)}}
    class_dict = {a: i for i, a in enumerate(activities)}
    class_dict[UNK] = len(class_dict)
    return Vocab(word_dict, class_dict)


def _repeat_next_edges(activity_ids: list[int]) -> tuple[list[int], list[int]]:
    """Ports RLHGNN's build_graph.get_index_of_duplicate_elements +
    the repeat-edge loop inside build_Bidirect_complex_graph. `activity_ids`
    is one prefix's encoded activity sequence in order; positions sharing
    the same id are treated as repeated occurrences of one activity (an id
    of 0, i.e. PAD, is never produced by `encode_graph` below, so no
    special-casing is needed here, unlike the original which explicitly
    skips value==0)."""
    positions_by_activity: dict[int, list[int]] = {}
    for pos, act in enumerate(activity_ids):
        positions_by_activity.setdefault(act, []).append(pos)

    n = len(activity_ids)
    src: list[int] = []
    dst: list[int] = []
    for positions in positions_by_activity.values():
        if len(positions) < 2:
            continue
        for i in range(len(positions) - 1):
            for j in range(i + 1, len(positions)):
                if positions[j] + 1 < n:
                    src.append(positions[i])
                    dst.append(positions[j] + 1)
        for i in range(len(positions) - 1, -1, -1):
            for j in range(i - 1, -1, -1):
                src.append(positions[i])
                dst.append(positions[j] + 1)
    return src, dst


def _build_graph(activity_ids: list[int]) -> dgl.DGLHeteroGraph:
    n = len(activity_ids)
    if n == 1:
        forward_src, forward_dst = [0], [0]
    else:
        forward_src, forward_dst = list(range(n - 1)), list(range(1, n))
    repeat_src, repeat_dst = _repeat_next_edges(activity_ids)

    graph = dgl.heterograph(
        {
            ("node", "forward", "node"): (forward_src, forward_dst),
            ("node", "backward", "node"): (forward_dst, forward_src),
            ("node", "repeat_next", "node"): (repeat_src, repeat_dst),
        }
    )
    graph.ndata["activity"] = torch.tensor(activity_ids, dtype=torch.long)
    return graph


def encode_graphs(prefix_df: pd.DataFrame, vocab: Vocab) -> tuple[list[dgl.DGLHeteroGraph], torch.Tensor]:
    graphs = [
        _build_graph([vocab.word_dict.get(tok, vocab.word_dict[UNK]) for tok in hist])
        for hist in prefix_df["history"]
    ]
    labels = torch.tensor(
        [vocab.class_dict.get(a, vocab.class_dict[UNK]) for a in prefix_df["next_act"]], dtype=torch.long
    )
    return graphs, labels


class PrefixGraphDataset(Dataset):
    """Minimal (graph, label) dataset compatible with
    dgl.dataloading.GraphDataLoader's default collate (dgl.batch + stack)."""

    def __init__(self, graphs: list[dgl.DGLHeteroGraph], labels: torch.Tensor):
        self.graphs = graphs
        self.labels = labels

    def __len__(self) -> int:
        return len(self.graphs)

    def __getitem__(self, idx: int) -> tuple[dgl.DGLHeteroGraph, torch.Tensor]:
        return self.graphs[idx], self.labels[idx]

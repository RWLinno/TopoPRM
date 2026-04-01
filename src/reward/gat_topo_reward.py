"""GAT-based topological reward scorer.

A lightweight Graph Attention Network that scores extracted reasoning DAGs.
Operates independently of LLM training — used as a drop-in replacement or
complement to the rule-based TopoReward.

Architecture:
    DAG -> Node features (step embedding + topo position encoding)
        -> GAT (2 layers, hidden=64, heads=4)
        -> Graph-level mean readout
        -> MLP head -> scalar score in [0, 1]
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from src.dag.graph import ReasoningDAG
from src.data.build_dag import build_dag_from_answer, extract_steps_from_answer
from src.reward.topo_position_encoding import compute_topo_position_encoding

# Optional torch import — module degrades gracefully if torch unavailable
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


# ---------------------------------------------------------------------------
# GAT layers (self-contained, no dependency on torch_geometric)
# ---------------------------------------------------------------------------

if _HAS_TORCH:

    class GATLayer(nn.Module):
        """Single-head graph attention layer."""

        def __init__(self, in_dim: int, out_dim: int, heads: int = 4, dropout: float = 0.1):
            super().__init__()
            self.heads = heads
            self.head_dim = out_dim // heads
            assert self.head_dim * heads == out_dim, "out_dim must be divisible by heads"

            self.W = nn.Linear(in_dim, out_dim, bias=False)
            self.a_src = nn.Parameter(torch.zeros(heads, self.head_dim))
            self.a_dst = nn.Parameter(torch.zeros(heads, self.head_dim))
            nn.init.xavier_uniform_(self.a_src.unsqueeze(0))
            nn.init.xavier_uniform_(self.a_dst.unsqueeze(0))
            self.dropout = nn.Dropout(dropout)
            self.leaky_relu = nn.LeakyReLU(0.2)

        def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
            """
            Args:
                x: (N, in_dim) node features
                adj: (N, N) adjacency matrix (0/1)
            Returns:
                (N, out_dim) updated node features
            """
            N = x.size(0)
            h = self.W(x).view(N, self.heads, self.head_dim)  # (N, H, D)

            # Attention scores
            e_src = (h * self.a_src.unsqueeze(0)).sum(-1)  # (N, H)
            e_dst = (h * self.a_dst.unsqueeze(0)).sum(-1)  # (N, H)
            attn = self.leaky_relu(
                e_src.unsqueeze(1) + e_dst.unsqueeze(2)  # (N, N, H) via broadcast
            ).permute(2, 0, 1)  # (H, N, N)

            # Mask non-edges with -inf
            mask = (adj.unsqueeze(0) == 0)  # (1, N, N)
            attn = attn.masked_fill(mask, float('-inf'))
            attn = F.softmax(attn, dim=-1)
            attn = attn.masked_fill(torch.isnan(attn), 0.0)
            attn = self.dropout(attn)

            # Aggregate
            h_t = h.permute(1, 0, 2)  # (H, N, D)
            out = torch.bmm(attn, h_t)  # (H, N, D)
            out = out.permute(1, 0, 2).reshape(N, -1)  # (N, H*D)
            return out

    class GATTopoScorer(nn.Module):
        """Two-layer GAT with graph-level readout for DAG quality scoring."""

        def __init__(
            self,
            input_dim: int = 19,  # 8 (lap PE) + 2 (degree) + 1 (depth) + 8 (step embed)
            hidden_dim: int = 64,
            heads: int = 4,
            dropout: float = 0.1,
        ):
            super().__init__()
            self.input_proj = nn.Linear(input_dim, hidden_dim)
            self.gat1 = GATLayer(hidden_dim, hidden_dim, heads=heads, dropout=dropout)
            self.gat2 = GATLayer(hidden_dim, hidden_dim, heads=heads, dropout=dropout)
            self.norm1 = nn.LayerNorm(hidden_dim)
            self.norm2 = nn.LayerNorm(hidden_dim)
            self.head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, 1),
                nn.Sigmoid(),
            )

        def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
            """Score a single DAG.

            Args:
                x: (N, input_dim) node features
                adj: (N, N) adjacency matrix
            Returns:
                scalar score in [0, 1]
            """
            h = F.relu(self.input_proj(x))
            h = self.norm1(F.relu(self.gat1(h, adj)) + h)
            h = self.norm2(F.relu(self.gat2(h, adj)) + h)
            # Graph-level mean readout
            graph_repr = h.mean(dim=0, keepdim=True)  # (1, hidden)
            score = self.head(graph_repr).squeeze()  # scalar
            return score


# ---------------------------------------------------------------------------
# Step embedding (lightweight, no pretrained model needed)
# ---------------------------------------------------------------------------

_MATH_PATTERN = re.compile(r'[=+\-*/^√∫∑∏<>≤≥≠≈]')
_NUM_PATTERN = re.compile(r'\d+\.?\d*')


def _step_features(step_text: str, max_dim: int = 8) -> np.ndarray:
    """Extract simple numerical features from a reasoning step."""
    feats = np.zeros(max_dim, dtype=np.float32)
    feats[0] = min(len(step_text) / 500.0, 1.0)  # normalised length
    feats[1] = min(len(_MATH_PATTERN.findall(step_text)) / 10.0, 1.0)  # math symbol density
    feats[2] = min(len(_NUM_PATTERN.findall(step_text)) / 10.0, 1.0)  # number density
    feats[3] = 1.0 if '=' in step_text else 0.0  # has equation
    feats[4] = 1.0 if any(kw in step_text for kw in ['因此', '所以', 'therefore', 'thus', 'hence']) else 0.0
    feats[5] = 1.0 if any(kw in step_text for kw in ['设', '令', 'let', 'define', 'assume']) else 0.0
    feats[6] = 1.0 if any(kw in step_text for kw in ['代入', 'substitute', 'plug']) else 0.0
    feats[7] = 1.0 if any(kw in step_text for kw in ['答', 'answer', '结论', 'conclusion']) else 0.0
    return feats


# ---------------------------------------------------------------------------
# ORM-compatible reward interface
# ---------------------------------------------------------------------------

@dataclass
class GATTopoConfig:
    """Configuration for GAT topo scorer."""
    pe_k: int = 8
    step_embed_dim: int = 8
    hidden_dim: int = 64
    heads: int = 4
    dropout: float = 0.1
    checkpoint_path: Optional[str] = None
    device: str = "cpu"


class GATTopoReward:
    """GAT-based topological reward, compatible with ms-swift ORM interface.

    Usage modes (controlled by TOPO_SCORER env var or config):
    - 'gat': pure GAT scoring
    - 'hybrid': 0.5 * rule_score + 0.5 * gat_score
    - 'rule_based': falls back to rule-based (default if torch unavailable)
    """

    def __init__(self, config: Optional[GATTopoConfig] = None, **kwargs: Any):
        self.config = config or GATTopoConfig()
        self._rule_based: Optional[Any] = None
        self._gat_model: Optional[Any] = None
        self._mode = os.environ.get("TOPO_SCORER", "rule_based")

        # Always initialise rule-based as fallback
        from src.reward.topo_reward import TopoReward
        self._rule_based = TopoReward()

        # Initialise GAT if requested and torch available
        if self._mode in ("gat", "hybrid") and _HAS_TORCH:
            input_dim = self.config.pe_k + 2 + 1 + self.config.step_embed_dim  # PE + degree + depth + step
            self._gat_model = GATTopoScorer(
                input_dim=input_dim,
                hidden_dim=self.config.hidden_dim,
                heads=self.config.heads,
                dropout=self.config.dropout,
            )
            if self.config.checkpoint_path and os.path.exists(self.config.checkpoint_path):
                state = torch.load(self.config.checkpoint_path, map_location=self.config.device)
                self._gat_model.load_state_dict(state)
            self._gat_model.to(self.config.device)
            self._gat_model.eval()

    def _score_dag_gat(self, trace: str) -> float:
        """Score a single trace using the GAT model."""
        if self._gat_model is None or not _HAS_TORCH:
            return 0.5  # neutral fallback

        steps = extract_steps_from_answer(trace)
        dag = build_dag_from_answer(trace)

        if dag.num_nodes == 0:
            return 0.0

        # Build node features
        edges = [(e.source, e.target) for e in dag.edges]
        topo_pe = compute_topo_position_encoding(
            dag.num_nodes, edges, k=self.config.pe_k
        )
        step_feats = np.stack([
            _step_features(s.content if hasattr(s, 'content') else str(s), self.config.step_embed_dim)
            for s in (dag.nodes if dag.nodes else steps[:dag.num_nodes])
        ])

        # Pad/truncate to match
        n = min(topo_pe.shape[0], step_feats.shape[0])
        node_features = np.concatenate([topo_pe[:n], step_feats[:n]], axis=1)

        # Build adjacency
        adj = np.zeros((n, n), dtype=np.float32)
        for u, v in edges:
            if 0 <= u < n and 0 <= v < n:
                adj[u, v] = 1.0
                adj[v, u] = 1.0
        # Add self-loops
        np.fill_diagonal(adj, 1.0)

        with torch.no_grad():
            x = torch.from_numpy(node_features).to(self.config.device)
            a = torch.from_numpy(adj).to(self.config.device)
            score = self._gat_model(x, a).item()

        return float(score)

    def __call__(
        self,
        completions: list,
        solution: Any = None,
        reference_dag: Any = None,
        **kwargs: Any,
    ) -> list[float]:
        if self._mode == "rule_based" or not _HAS_TORCH:
            return self._rule_based(completions, solution=solution, reference_dag=reference_dag, **kwargs)

        if self._mode == "gat":
            rewards = []
            for completion in completions:
                text = completion if isinstance(completion, str) else (
                    completion[-1].get("content", "") if completion else ""
                )
                rewards.append(self._score_dag_gat(text))
            return rewards

        # hybrid mode
        rule_scores = self._rule_based(completions, solution=solution, reference_dag=reference_dag, **kwargs)
        gat_scores = []
        for completion in completions:
            text = completion if isinstance(completion, str) else (
                completion[-1].get("content", "") if completion else ""
            )
            gat_scores.append(self._score_dag_gat(text))

        hybrid_weight = float(os.environ.get("TOPO_HYBRID_WEIGHT", "0.5"))
        return [
            round(hybrid_weight * g + (1 - hybrid_weight) * r, 6)
            for g, r in zip(gat_scores, rule_scores)
        ]


# Register in ms-swift orms if available
try:
    from swift.rewards import orms
    orms["gat_topo_reward"] = GATTopoReward
except ImportError:
    pass

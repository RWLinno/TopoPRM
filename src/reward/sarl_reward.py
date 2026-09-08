"""Protocol-matched SARL small-world topology reward baseline.

This reimplements the public SARL reward described by Wang et al. (2026):
reasoning steps are embedded, clustered into latent functions, linked by
consecutive transitions, and scored by clustering and shortest-path terms.
It is intentionally separate from TopoPRM's semantic-support graph reward.
"""

from __future__ import annotations

import math
import os
import re
from typing import Any, Sequence

import networkx as nx
import numpy as np

from swift.rewards import ORM, orms

from src.dag.edge_encoder import FrozenStepEmbedder
from src.reward.utils import completion_to_text, extract_think_block


_MIN_STEP_LENGTH = 20
_STEP_MARKER_RE = re.compile(
    r"(?:^|\n)(?:"
    r"Wait[,. ]|Hmm[,. ]|Actually[,. ]|No[,. ]|"
    r"Let me (?:re|try|check|verify|reconsider|re-examine|think)|"
    r"I (?:made an error|was wrong|need to reconsider)|"
    r"Alternatively[,. ]|Another (?:approach|way|method)|"
    r"Checking:|Indeed[,. ]|This confirms|So the answer|The answer is|"
    r"Step \d+[:.)]|First[,. ]|Second[,. ]|Third[,. ]|Next[,. ]|"
    r"Finally[,. ]|Therefore[,. ]|Thus[,. ]|Hence[,. ]|So[,. ]"
    r")",
    re.MULTILINE | re.IGNORECASE,
)


def _extract_steps(text: str) -> list[str]:
    """Match the public SARL paragraph/marker/newline segmentation order."""
    text = text.strip()
    if not text:
        return []
    paragraphs = [part.strip() for part in re.split(r"\n{2,}", text) if part.strip()]
    steps: list[str] = []
    for paragraph in paragraphs:
        if len(paragraph) > 200:
            boundaries = [match.start() for match in _STEP_MARKER_RE.finditer(paragraph)]
            if boundaries:
                fragments: list[str] = []
                previous = 0
                for boundary in boundaries:
                    fragment = paragraph[previous:boundary].strip()
                    if fragment:
                        fragments.append(fragment)
                    previous = boundary
                tail = paragraph[previous:].strip()
                if tail:
                    fragments.append(tail)
                if len(fragments) > 1:
                    steps.extend(
                        item for item in fragments if len(item) >= _MIN_STEP_LENGTH
                    )
                    continue
        if len(paragraph) >= _MIN_STEP_LENGTH:
            steps.append(paragraph)
    if not steps:
        steps = [
            line.strip()
            for line in text.splitlines()
            if len(line.strip()) >= _MIN_STEP_LENGTH
        ]
    return steps


def _cluster_transitions(
    embeddings: np.ndarray,
    cluster_method: str = "hdbscan",
) -> nx.Graph:
    """Build the official SARL undirected, weighted transition graph."""
    from sklearn.cluster import KMeans

    count = len(embeddings)
    graph = nx.Graph()
    if count <= 1:
        return graph

    if cluster_method == "hdbscan":
        import hdbscan

        min_cluster_size = max(2, min(5, count // 4)) if count >= 4 else 2
        labels = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=max(1, min_cluster_size - 1),
            metric="euclidean",
            cluster_selection_method="eom",
        ).fit_predict(embeddings)
        noise = labels == -1
        if noise.any():
            labels[noise] = 0 if noise.all() else labels.max() + 1
    elif cluster_method == "kmeans":
        clusters = min(max(2, int(math.sqrt(count))), count)
        labels = KMeans(
            n_clusters=clusters,
            n_init=10,
            random_state=42,
        ).fit_predict(embeddings)
    else:
        raise ValueError(f"Unsupported SARL cluster method: {cluster_method}")

    distances = [
        float(np.linalg.norm(embeddings[index] - embeddings[index + 1]))
        for index in range(count - 1)
    ]
    for left, right, distance in zip(labels[:-1], labels[1:], distances):
        source, target = int(left), int(right)
        if source != target:
            if graph.has_edge(source, target):
                edge = graph[source][target]
                previous_count = int(edge.get("count", 1))
                edge["weight"] = (
                    float(edge["weight"]) * previous_count + distance
                ) / (previous_count + 1)
                edge["count"] = previous_count + 1
            else:
                graph.add_edge(source, target, weight=distance, count=1)
    return graph


def _score_graph(graph: nx.Graph) -> float:
    if graph.number_of_nodes() <= 1:
        return 0.0
    clustering = float(nx.average_clustering(graph))
    try:
        path_term = min(
            1.0 / (1.0 + float(nx.average_shortest_path_length(graph))),
            0.5,
        )
    except (nx.NetworkXError, nx.NetworkXPointlessConcept):
        path_term = 0.0
    return max(0.0, min(1.0, 0.5 * clustering + path_term))


class SARLStructureReward(ORM):
    """Matched small-world topology baseline using the released 0.6B encoder."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__()
        model_path = os.environ.get(
            "SARL_EMBED_MODEL",
            "/knowin-oss/weilinruan/models/Qwen3-Embedding-0.6B",
        )
        if not os.path.isdir(model_path):
            raise FileNotFoundError(f"SARL embedding model not found: {model_path}")
        self._embedder = FrozenStepEmbedder(
            model_path=model_path,
            device=os.environ.get("SARL_EMBED_DEVICE", "auto"),
            batch_size=int(os.environ.get("SARL_EMBED_BATCH_SIZE", "8")),
            max_length=int(os.environ.get("SARL_EMBED_MAX_LENGTH", "4096")),
            instruction="",
            dtype=os.environ.get("SARL_EMBED_DTYPE", "bfloat16"),
        )
        self._cluster_method = os.environ.get("SARL_CLUSTER_METHOD", "hdbscan").lower()
        if self._cluster_method not in {"hdbscan", "kmeans"}:
            raise ValueError(f"Unsupported SARL cluster method: {self._cluster_method}")
        self._num_calls = 0
        self._log_every = max(0, int(os.environ.get("SARL_REWARD_LOG_EVERY", "10")))

    def __call__(
        self,
        completions: Sequence[Any],
        **kwargs: Any,
    ) -> list[float]:
        steps_by_completion: list[list[str]] = []
        for completion in completions:
            think_text = extract_think_block(completion_to_text(completion))
            steps_by_completion.append(_extract_steps(think_text) if think_text else [])

        flat_steps = [step for steps in steps_by_completion for step in steps]
        if flat_steps:
            encoded = self._embedder.encode(flat_steps).numpy().astype(np.float32, copy=False)
        else:
            encoded = np.empty((0, 0), dtype=np.float32)

        rewards: list[float] = []
        offset = 0
        for steps in steps_by_completion:
            next_offset = offset + len(steps)
            graph = _cluster_transitions(
                encoded[offset:next_offset],
                cluster_method=self._cluster_method,
            )
            rewards.append(_score_graph(graph))
            offset = next_offset

        self._num_calls += 1
        if self._log_every and self._num_calls % self._log_every == 0 and rewards:
            print(
                "[sarl_structure] "
                f"n={len(rewards)} mean={sum(rewards) / len(rewards):.4f} "
                f"min={min(rewards):.4f} max={max(rewards):.4f}"
            )
        return rewards


orms["sarl_structure"] = SARLStructureReward

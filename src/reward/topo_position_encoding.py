"""Topological position encoding for DAG nodes.

Provides Laplacian eigenvector-based position features that capture
global structural information in the extracted reasoning DAG.
"""

from __future__ import annotations

import numpy as np
from typing import Optional


def _adjacency_matrix(num_nodes: int, edges: list[tuple[int, int]]) -> np.ndarray:
    """Build adjacency matrix from edge list."""
    A = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    for u, v in edges:
        if 0 <= u < num_nodes and 0 <= v < num_nodes:
            A[u, v] = 1.0
            A[v, u] = 1.0  # undirected for Laplacian
    return A


def _laplacian_eigenvectors(A: np.ndarray, k: int = 8) -> np.ndarray:
    """Compute top-k smallest non-trivial Laplacian eigenvectors.

    Returns shape (num_nodes, k). If the graph has fewer than k+1 nodes,
    zero-pads the remaining dimensions.
    """
    n = A.shape[0]
    if n == 0:
        return np.zeros((0, k), dtype=np.float32)

    D = np.diag(A.sum(axis=1))
    L = D - A

    try:
        eigenvalues, eigenvectors = np.linalg.eigh(L)
    except np.linalg.LinAlgError:
        return np.zeros((n, k), dtype=np.float32)

    # Skip the first eigenvector (constant, eigenvalue ~0)
    start = 1
    end = min(start + k, n)
    selected = eigenvectors[:, start:end]

    # Zero-pad if fewer eigenvectors than k
    if selected.shape[1] < k:
        pad = np.zeros((n, k - selected.shape[1]), dtype=np.float32)
        selected = np.concatenate([selected, pad], axis=1)

    return selected.astype(np.float32)


def compute_topo_position_encoding(
    num_nodes: int,
    edges: list[tuple[int, int]],
    k: int = 8,
    include_degree: bool = True,
    include_depth: bool = True,
) -> np.ndarray:
    """Compute topological position encoding for each node.

    Returns shape (num_nodes, feat_dim) where feat_dim = k + extras.

    Features per node:
    - Laplacian eigenvector components (k dims)
    - Normalised in-degree (1 dim, if include_degree)
    - Normalised out-degree (1 dim, if include_degree)
    - Normalised topological depth (1 dim, if include_depth)
    """
    if num_nodes == 0:
        extra = int(include_degree) * 2 + int(include_depth)
        return np.zeros((0, k + extra), dtype=np.float32)

    A = _adjacency_matrix(num_nodes, edges)
    lap_pe = _laplacian_eigenvectors(A, k=k)

    features = [lap_pe]

    if include_degree:
        # Directed degree from original edges
        in_deg = np.zeros(num_nodes, dtype=np.float32)
        out_deg = np.zeros(num_nodes, dtype=np.float32)
        for u, v in edges:
            if 0 <= u < num_nodes and 0 <= v < num_nodes:
                out_deg[u] += 1
                in_deg[v] += 1
        max_deg = max(in_deg.max(), out_deg.max(), 1.0)
        features.append((in_deg / max_deg).reshape(-1, 1))
        features.append((out_deg / max_deg).reshape(-1, 1))

    if include_depth:
        depth = _topological_depth(num_nodes, edges)
        max_depth = max(depth.max(), 1.0)
        features.append((depth / max_depth).reshape(-1, 1))

    return np.concatenate(features, axis=1)


def _topological_depth(num_nodes: int, edges: list[tuple[int, int]]) -> np.ndarray:
    """Compute longest-path depth for each node via topological sort."""
    depth = np.zeros(num_nodes, dtype=np.float32)
    adj: dict[int, list[int]] = {i: [] for i in range(num_nodes)}
    in_deg = np.zeros(num_nodes, dtype=int)

    for u, v in edges:
        if 0 <= u < num_nodes and 0 <= v < num_nodes:
            adj[u].append(v)
            in_deg[v] += 1

    # Kahn's algorithm
    queue = [i for i in range(num_nodes) if in_deg[i] == 0]
    while queue:
        next_queue = []
        for u in queue:
            for v in adj[u]:
                depth[v] = max(depth[v], depth[u] + 1)
                in_deg[v] -= 1
                if in_deg[v] == 0:
                    next_queue.append(v)
        queue = next_queue

    return depth

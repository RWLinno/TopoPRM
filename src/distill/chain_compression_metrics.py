from __future__ import annotations

from src.data.build_dag import build_dag_from_answer, extract_steps_from_answer


def chain_compression_metrics(trace: str) -> dict[str, float]:
    dag = build_dag_from_answer(trace)
    return {
        "num_steps": float(len(extract_steps_from_answer(trace))),
        "num_nodes": float(dag.num_nodes),
        "num_edges": float(dag.num_edges),
        "dep_depth": float(dag.get_dependency_depth()),
        "num_chars": float(len(trace)),
    }
